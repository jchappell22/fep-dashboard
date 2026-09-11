"""fepdash.core.driver -- render the bash script a campaign actually runs.

Why a generated shell script instead of a Python orchestrator
-------------------------------------------------------------

The dashboard is written on a laptop and run on GPU2, against engine
versions nobody here can test. When something goes wrong at hour 9 of a
campaign, the person debugging it needs to be able to *read the thing that
ran* and re-run one piece of it by hand. A generated ``driver.sh`` gives
them that:

* the exact command lines are on disk, in the campaign directory;
* it is resumable -- finished legs are skipped, so re-running after a crash
  picks up where it stopped;
* if a rendered flag turns out to be wrong, they can edit ``driver.sh`` and
  ``bash driver.sh`` it directly, with no dashboard involved;
* Streamlit crashing, restarting, or being Ctrl-C'd cannot affect a run,
  because nothing about the run lives in the Streamlit process.

The dashboard spawns this script detached (``start_new_session=True``) and
afterwards only ever *reads* the campaign directory.

Two shapes of run stage
-----------------------

``per_gpu`` (OpenFE)
    The engine has no GPU scheduler of its own, so the driver fans legs out:
    one worker per selected GPU, each pinned with ``CUDA_VISIBLE_DEVICES``,
    all pulling from a shared queue via atomic ``mkdir`` claims. Fast solvent
    legs therefore don't leave a GPU idle behind a slow complex leg.

``single`` (TMD)
    The engine schedules its own work across GPUs through the CUDA MPS
    daemon. The driver must *not* fan out -- it runs one process and exposes
    the whole GPU set to it. Fanning out here would oversubscribe the cards
    and is the most likely way to make two engines fight.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .engines.base import Engine, build_variables, render
from .models import Campaign


_HEADER = r"""#!/usr/bin/env bash
# =========================================================================
# fep-dashboard driver -- GENERATED FILE, safe to read, safe to edit.
#
#   campaign : {campaign_id}  ({name})
#   engine   : {engine_name}  ({method})
#   gpus     : {gpu_list}
#   created  : {created_at}
#
# Re-running this script is safe and resumes: completed legs are skipped.
# To retry only the failed legs, delete their directories under legs/ and
# re-run.
# =========================================================================
set -uo pipefail

CAMPAIGN_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
cd "$CAMPAIGN_DIR" || exit 1

mkdir -p plan legs work results gathered

stage() {{ printf '%s\n' "$1" > "$CAMPAIGN_DIR/.stage"; }}
log()   {{ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }}

# Write one leg's status atomically: a half-written status.json read by the
# dashboard mid-write would show a leg in a state it was never in.
leg_status() {{  # leg_status <leg> <status> [key=value ...]
    local leg="$1" st="$2"; shift 2
    local dir="$CAMPAIGN_DIR/legs/$leg"
    mkdir -p "$dir"
    {{
        printf '{{"leg_id": "%s", "status": "%s"' "$leg" "$st"
        for kv in "$@"; do
            printf ', "%s": "%s"' "${{kv%%=*}}" "${{kv#*=}}"
        done
        printf '}}\n'
    }} > "$dir/status.json.tmp"
    mv -f "$dir/status.json.tmp" "$dir/status.json"
}}

fail() {{ log "FAILED: $*"; stage done; exit 1; }}
"""


# Both engines on Conifer are reached through launchers in /usr/local/bin
# (`openfe`, `tmd-submit`) that activate their own shared /opt conda env
# internally. So the normal driver does no conda handling at all -- it just
# checks the launcher is there.
_ENV_BLOCK = r"""
# ---- engine environment -------------------------------------------------
{exports}
command -v {binary} >/dev/null 2>&1 || fail "'{binary}' not on PATH -- expected the shared launcher in /usr/local/bin"
log "engine launcher: $(command -v {binary})"
"""

# Retained for an engine that is only available inside a personal conda env
# rather than behind a shared launcher. The hook form covers non-interactive
# shells (nohup/setsid), where `conda activate` alone does not work.
_ENV_BLOCK_CONDA = r"""
# ---- engine environment -------------------------------------------------
{exports}
if ! command -v {binary} >/dev/null 2>&1; then
    eval "$(mamba shell hook --shell bash 2>/dev/null)" 2>/dev/null || true
    source "{conda_root}/etc/profile.d/conda.sh" 2>/dev/null || true
    mamba activate "{conda_env}" 2>/dev/null || conda activate "{conda_env}" 2>/dev/null || true
fi
command -v {binary} >/dev/null 2>&1 || fail "'{binary}' not on PATH after activating '{conda_env}' -- check [env] in the engine TOML"
log "engine binary: $(command -v {binary})"
"""


_PLAN_BLOCK = r"""
# ---- stage: plan --------------------------------------------------------
stage plan
if [ -e "$CAMPAIGN_DIR/.plan_done" ]; then
    log "plan already complete, skipping"
else
    log "planning: {stage_label}"
{plan_cmds}
    touch "$CAMPAIGN_DIR/.plan_done"
    log "plan complete"
fi
"""


_RUN_PER_GPU = r"""
# ---- stage: run (fan out one leg per GPU) -------------------------------
stage run
GPUS="${{GPUS:-{gpu_list}}}"
shopt -s nullglob
JOBS=({jobs_glob})
shopt -u nullglob
log "run: ${{#JOBS[@]}} legs across GPUs [$GPUS]"
[ "${{#JOBS[@]}}" -gt 0 ] || fail "planning produced no job files under plan/{jobs_glob_display}"

# Release claims left behind by a killed run: a leg claimed but with no
# result was interrupted, not finished. Without this, a crashed campaign
# can never be resumed -- every leg looks taken.
for d in "$CAMPAIGN_DIR"/legs/*/; do
    [ -d "$d" ] || continue
    name="$(basename "$d")"
    if [ -e "$d/.claim" ] && [ ! -e "$CAMPAIGN_DIR/{done_marker_glob}" ]; then
        case "$(cat "$d/status.json" 2>/dev/null)" in
            *'"status": "done"'*) ;;
            *) rm -rf "$d/.claim"; log "released stale claim: $name" ;;
        esac
    fi
done

worker() {{
    local gpu="$1"
    export CUDA_VISIBLE_DEVICES="$gpu"
    for tf in "${{JOBS[@]}}"; do
        local name; name="$(basename "$tf" .json)"
        local legdir="$CAMPAIGN_DIR/legs/$name"
        # Already finished (this run or a previous one)?
        case "$(cat "$legdir/status.json" 2>/dev/null)" in
            *'"status": "done"'*) continue ;;
        esac
        mkdir -p "$legdir"
        # Atomic claim. mkdir fails if another worker got here first, which
        # is what keeps two GPUs off the same leg without a lock file.
        mkdir "$legdir/.claim" 2>/dev/null || continue

        local t0; t0="$(date +%s)"
        leg_status "$name" running "gpu=$gpu" "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        log "[gpu $gpu] start $name"
        if {run_cmd} > "$legdir/leg.log" 2>&1; then
            leg_status "$name" done "gpu=$gpu" \
                "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
                "duration_s=$(( $(date +%s) - t0 ))"
            log "[gpu $gpu] done  $name"
        else
            leg_status "$name" failed "gpu=$gpu" \
                "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
                "duration_s=$(( $(date +%s) - t0 ))"
            # Do NOT release the claim here. A leg that just failed will
            # almost always fail the same way immediately, and releasing it
            # mid-run lets every other idle worker pick it up and fail on it
            # in turn -- burning the whole box on one doomed leg. The claim
            # is released by the stale-claim sweep at the START of the next
            # run instead, so retrying is an explicit act (Resume), not an
            # accident.
            log "[gpu $gpu] FAIL  $name (legs/$name/leg.log)"
        fi
    done
    log "[gpu $gpu] queue empty"
}}

pids=()
for g in $GPUS; do worker "$g" & pids+=($!); done
wait "${{pids[@]}}"
log "run stage finished"
"""


_RUN_SINGLE = r"""
# ---- stage: run (single process; the engine schedules its own work) -----
stage run
{gpu_export}LEG=graph
log "run: single-process engine ({gpu_note})"
case "$(cat "$CAMPAIGN_DIR/legs/$LEG/status.json" 2>/dev/null)" in
    *'"status": "done"'*) log "already complete, skipping"; SKIP=1 ;;
    *) SKIP=0 ;;
esac
if [ "$SKIP" = "0" ]; then
    t0="$(date +%s)"
    leg_status "$LEG" running "gpu={gpu_csv}" "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    mkdir -p "$CAMPAIGN_DIR/legs/$LEG"
    if {run_cmd} > "$CAMPAIGN_DIR/legs/$LEG/leg.log" 2>&1; then
        leg_status "$LEG" done "gpu={gpu_csv}" \
            "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            "duration_s=$(( $(date +%s) - t0 ))"
        log "run complete"
    else
        leg_status "$LEG" failed "gpu={gpu_csv}" \
            "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            "duration_s=$(( $(date +%s) - t0 ))"
        log "run FAILED (legs/$LEG/leg.log)"
    fi
fi
"""


_GATHER_BLOCK = r"""
# ---- stage: gather ------------------------------------------------------
stage gather
log "gathering results"
{gather_cmds}
"""

_GATHER_NONE = r"""
# ---- stage: gather ------------------------------------------------------
stage gather
# This engine writes its own result tables as it runs; nothing to gather.
log "engine writes its own tables -- no gather step"
"""

_FOOTER = r"""
stage done
log "campaign finished"
"""


def render_driver(
    campaign: Campaign,
    engine: Engine,
    *,
    python: str = "python",
    scripts_dir: Optional[Path] = None,
) -> str:
    """Produce the complete ``driver.sh`` text for one campaign."""
    variables = build_variables(campaign, engine, python=python, scripts_dir=scripts_dir)
    method = campaign.method
    gpu_list = " ".join(str(g) for g in campaign.gpus) or "0"
    gpu_csv = ",".join(str(g) for g in campaign.gpus) or "0"

    parts = [
        _HEADER.format(
            campaign_id=campaign.campaign_id,
            name=campaign.name,
            engine_name=engine.name,
            method=method.value,
            gpu_list=gpu_list if not engine.picks_own_gpu else "chosen by the engine's launcher",
            created_at=campaign.created_at,
        )
    ]

    # -- environment ------------------------------------------------------
    export_lines = [f'export {k}="{v}"' for k, v in engine.exports.items()]

    # An engine version pin, where the launcher supports one. OpenFE's
    # /opt launcher reads OPENFE_VERSION to choose between the installed
    # 1.8.1 / 1.11.1 / 1.12.0 envs; unset means /opt/openfe/current.
    version = str(campaign.params.get("engine_version", "") or "")
    if version and engine.version_env_var:
        export_lines.append(f'export {engine.version_env_var}="{version}"')

    exports = "\n".join(export_lines)
    if engine.engine_workdir:
        # For an engine invoked as `python examples/...` rather than through
        # a launcher, the commands only resolve from inside its checkout.
        # Everything downstream uses absolute paths, so cd'ing is safe.
        exports += f'\ncd "{engine.engine_workdir}" || fail "engine workdir not found: {engine.engine_workdir}"'

    env_block = _ENV_BLOCK_CONDA if engine.needs_conda else _ENV_BLOCK
    parts.append(
        env_block.format(
            exports=exports,
            binary=engine.binary or "python",
            conda_root=engine.conda_root or "$HOME/miniconda3",
            conda_env=engine.conda_env or "base",
        )
    )

    # -- plan -------------------------------------------------------------
    plan_spec = engine.stage(method, "plan")
    plan_cmds = _render_cmds(plan_spec.cmds, variables, engine, method, "plan", indent=4)
    if plan_cmds.strip():
        parts.append(
            _PLAN_BLOCK.format(
                stage_label=plan_spec.get("stage_label", "plan"),
                plan_cmds=plan_cmds,
            )
        )

    # -- run --------------------------------------------------------------
    run_spec = engine.stage(method, "run")
    if not run_spec.cmds:
        raise ValueError(f"engine '{engine.name}' has no [{method.value}.run] cmd")
    run_cmd = render(
        run_spec.cmds[0], variables, where=f"{engine.name}/{method.value}.run.cmd"
    )

    if engine.parallel_mode(method) == "single":
        if engine.picks_own_gpu:
            # Do NOT pin cards. tmd-submit picks the most-free GPU on a
            # shared box; exporting CUDA_VISIBLE_DEVICES here would override
            # that and start stepping on other people's jobs.
            gpu_export = (
                "# GPU choice is left to the engine's own launcher, which picks\n"
                "# the most-free card. Setting CUDA_VISIBLE_DEVICES here would\n"
                "# override that and defeat sharing on this box.\n"
            )
            gpu_note = "GPU chosen by the launcher"
            gpu_csv = "auto"
        else:
            gpu_export = f'export CUDA_VISIBLE_DEVICES="{gpu_csv}"\n'
            gpu_note = f"GPUs [{gpu_csv}]"
        parts.append(
            _RUN_SINGLE.format(
                run_cmd=run_cmd,
                gpu_csv=gpu_csv,
                gpu_export=gpu_export,
                gpu_note=gpu_note,
            )
        )
    else:
        jobs_glob = plan_spec.get("jobs_glob", "")
        if not jobs_glob:
            raise ValueError(
                f"engine '{engine.name}' fans out per GPU but declares no "
                f"jobs_glob in [{method.value}.plan]"
            )
        done_marker = run_spec.get("done_marker", "results/$name.json")
        parts.append(
            _RUN_PER_GPU.format(
                gpu_list=gpu_list,
                jobs_glob=f'"$CAMPAIGN_DIR"/plan/{jobs_glob}',
                jobs_glob_display=jobs_glob,
                run_cmd=run_cmd,
                done_marker_glob=done_marker.replace("{leg}", "$name"),
            )
        )

    # -- gather -----------------------------------------------------------
    gather_spec = engine.stage(method, "gather")
    if gather_spec.cmds:
        # A gather failure must not mark the campaign failed: with OpenFE's
        # `gather --report dg` it usually just means "fewer than two repeats
        # so far", which is expected mid-campaign.
        lines = []
        for i, tmpl in enumerate(gather_spec.cmds):
            text = render(
                tmpl, variables, where=f"{engine.name}/{method.value}.gather.cmd[{i}]"
            )
            lines.append(f'{text} || log "gather step {i} returned non-zero (often means results are still partial)"')
        parts.append(_GATHER_BLOCK.format(gather_cmds="\n".join(lines)))
    else:
        parts.append(_GATHER_NONE)

    parts.append(_FOOTER)
    return "".join(parts)


def _render_cmds(
    cmds: tuple[str, ...],
    variables: dict,
    engine: Engine,
    method,
    stage: str,
    *,
    indent: int = 0,
) -> str:
    pad = " " * indent
    out = []
    for i, tmpl in enumerate(cmds):
        text = render(tmpl, variables, where=f"{engine.name}/{method.value}.{stage}.cmd[{i}]")
        if not text:
            continue
        out.append(f"{pad}{text} || fail \"{stage} step {i} failed\"")
    return "\n".join(out)


def write_driver(
    campaign: Campaign,
    engine: Engine,
    *,
    python: str = "python",
    scripts_dir: Optional[Path] = None,
) -> Path:
    """Render and write ``driver.sh``, mode 0755."""
    text = render_driver(campaign, engine, python=python, scripts_dir=scripts_dir)
    path = campaign.driver_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)
    return path
