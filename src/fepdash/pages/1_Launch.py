"""Launch page -- build a campaign, read the command, then commit the GPUs.

The design rule here: **nothing is spawned until the operator has seen the
literal command line.** The engine TOMLs are written against documentation
for TMD and against a possibly-different openfe version for OpenFE, so the
rendered command is the only honest preview of what will run. Committing a
card for 12 hours behind a hidden argv is how days get lost.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fepdash.core import launcher  # noqa: E402
from fepdash.core.config import load_config  # noqa: E402
from fepdash.core.driver import render_driver  # noqa: E402
from fepdash.core.engines.base import (  # noqa: E402
    TemplateError,
    load_engines,
    preview_commands,
)
from fepdash.core.models import Campaign, Method, new_campaign_id  # noqa: E402
from fepdash.ui.common import gpu_board, gpu_selector, page_header  # noqa: E402

st.set_page_config(page_title="Launch — FEP dashboard", page_icon="🚀", layout="wide")


def _referenced_variables(engine, method) -> set[str]:
    """Placeholder names this engine's templates actually use for a method.

    Drives which parameter widgets appear, so the form shows the knobs that
    do something and hides the ones that do not. An engine whose launcher
    supplies its own defaults (TMD) correctly ends up with an empty form.
    """
    import string

    names: set[str] = set()
    for stage in ("plan", "run", "gather"):
        try:
            spec = engine.stage(method, stage)
        except Exception:
            continue
        for template in spec.cmds:
            names.update(
                field for _, field, _, _ in string.Formatter().parse(template) if field
            )
    return names

cfg = load_config()
engines = load_engines(cfg.engines_dir)
usable = {name: e for name, e in engines.items() if e.methods}

page_header("Launch a campaign", "Protein + ligands + engine → GPU work.")

if not usable:
    st.error(f"No usable engine definitions in {cfg.engines_dir}.")
    st.stop()


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Engine")
    engine_name = st.selectbox(
        "Engine", options=list(usable), format_func=lambda n: usable[n].label
    )
    engine = usable[engine_name]

    method_value = st.selectbox(
        "Method",
        options=list(engine.methods),
        format_func=lambda m: Method(m).label,
    )
    method = Method(method_value)

    # Version pin, where the engine's launcher supports one. OpenFE's /opt
    # launcher reads OPENFE_VERSION; blank means /opt/openfe/current.
    engine_version = ""
    if engine.available_versions:
        engine_version = st.selectbox(
            "Engine version",
            options=engine.available_versions,
            format_func=lambda v: v or "default (current symlink)",
            help=(
                f"Exported as ${engine.version_env_var} for this campaign. "
                f"Pin an older build to reproduce a previous result; leave on "
                f"default for new work."
            ),
        )

    name = st.text_input("Campaign name", value="", placeholder="tyk2 congeneric series")

with col_right:
    st.subheader("Inputs")
    st.caption(
        f"Paths are resolved on **this box**. Browse from `{cfg.inputs_root}` or "
        f"paste an absolute path."
    )
    protein = st.text_input("Protein (PDB or mmCIF)", value="", placeholder="/home/jacob/inputs/tyk2.pdb")
    ligands = st.text_input("Ligands (SDF)", value="", placeholder="/home/jacob/inputs/ligands.sdf")

    with st.expander("Optional inputs"):
        cofactors = st.text_input("Cofactors SDF", value="")
        settings_yaml = st.text_input("Settings YAML (mapper / network / charges)", value="")


st.subheader("GPUs")
if engine.picks_own_gpu:
    # No picker: `tmd-submit` chooses the most-free card itself. Offering a
    # control the engine ignores would be worse than offering none.
    gpus = []
    st.info(
        f"**{engine.name}** picks its own GPU — its launcher selects the "
        f"most-free card at submit time. The dashboard deliberately does not "
        f"pin `CUDA_VISIBLE_DEVICES` for it, so the box stays shareable."
    )
    gpu_board(cfg)
    st.caption(
        "Shown for awareness only. This box is shared — sanity-check "
        "`nvidia-smi` before committing a long run, and never kill foreign jobs."
    )
else:
    gpus = gpu_selector(cfg, key="launch_gpus")
    st.caption(
        f"**{engine.name}** has no GPU scheduler, so the driver runs one leg per "
        f"card, pulling from a shared queue. More cards = proportionally faster."
    )


# ---------------------------------------------------------------------------
# Engine parameters
# ---------------------------------------------------------------------------

st.subheader("Parameters")
params: dict = {"engine_version": engine_version}
defaults = engine.defaults

# Which knobs this engine's templates actually reference. Rendering a widget
# for a variable no command uses just invites someone to set it and wonder
# why nothing changed.
referenced = _referenced_variables(engine, method)

pcols = st.columns(4)
slot = 0

if "n_repeats" in referenced:
    with pcols[slot % 4]:
        params["n_repeats"] = st.number_input(
            "protocol repeats / edge",
            min_value=1, max_value=10,
            value=int(defaults.get("n_repeats", 3)),
            help=(
                "OpenFE's per-ligand ΔG is a maximum-likelihood estimate across "
                "repeats and needs at least 2. Per-edge ΔΔG works with 1."
            ),
        )
    slot += 1

if "n_cores" in referenced:
    with pcols[slot % 4]:
        params["n_cores"] = st.number_input(
            "CPU cores (planning)", min_value=1, max_value=128,
            value=int(defaults.get("n_cores", 8)),
            help="Partial-charge generation during planning is CPU-bound, not GPU-bound.",
        )
    slot += 1

# A single-edge run needs the two ligand names, and they must match the
# names inside the SDF exactly -- that is what the engine looks them up by.
if method is Method.EDGE:
    with pcols[slot % 4]:
        params["ligand_a"] = st.text_input("ligand A", value="", placeholder="ejm_31")
    with pcols[(slot + 1) % 4]:
        params["ligand_b"] = st.text_input("ligand B", value="", placeholder="ejm_46")
    slot += 2
    st.caption("Names must match the molecule titles inside the SDF exactly.")

# Anything else the engine TOML declares and its templates reference.
handled = {"n_repeats", "n_cores", "ligand_a", "ligand_b", "engine_version"}
for key in sorted(referenced - handled):
    if key not in defaults:
        continue
    with pcols[slot % 4]:
        default = defaults[key]
        if isinstance(default, bool):
            params[key] = st.checkbox(key, value=default)
        elif isinstance(default, int):
            params[key] = st.number_input(key, value=int(default), step=1)
        elif isinstance(default, float):
            params[key] = st.number_input(key, value=float(default))
        else:
            params[key] = st.text_input(key, value=str(default))
    slot += 1

if slot == 0:
    st.caption(
        f"`{engine.name}` takes no tunable parameters for this method — its "
        f"launcher supplies its own defaults (including the forcefield)."
    )


# ---------------------------------------------------------------------------
# Build the campaign object and preview
# ---------------------------------------------------------------------------

campaign = Campaign(
    campaign_id=new_campaign_id(name or f"{engine_name}-{method.value}"),
    name=name or f"{engine_name} {method.value}",
    engine=engine_name,
    method=method,
    protein=Path(protein) if protein else Path(""),
    ligands=Path(ligands) if ligands else Path(""),
    cofactors=Path(cofactors) if cofactors else None,
    settings_yaml=Path(settings_yaml) if settings_yaml else None,
    gpus=[int(g) for g in gpus],
    params=params,
    run_dir=cfg.runs_root / "PREVIEW",
)

st.subheader("What will run")

try:
    preview = preview_commands(
        campaign, engine, python="python", scripts_dir=cfg.scripts_dir
    )
except TemplateError as exc:
    st.error(f"Engine template problem: {exc}")
    st.stop()

for stage_name in ("plan", "run", "gather"):
    cmds = preview.get(stage_name) or []
    if not cmds:
        st.markdown(f"**{stage_name}** — _nothing to do for this engine_")
        continue
    st.markdown(f"**{stage_name}**")
    for cmd in cmds:
        st.code(cmd, language="bash")

st.caption(
    "`$tf` and `$name` are substituted per leg by the driver script. "
    "`PREVIEW` in the paths becomes the real campaign id on launch."
)

with st.expander("Full driver.sh"):
    try:
        st.code(
            render_driver(campaign, engine, scripts_dir=cfg.scripts_dir),
            language="bash",
        )
    except Exception as exc:  # noqa: BLE001 - preview must never hard-fail
        st.error(f"could not render driver: {exc}")


# ---------------------------------------------------------------------------
# Pre-flight and launch
# ---------------------------------------------------------------------------

st.subheader("Pre-flight")
problems = launcher.preflight(campaign, engine, cfg)
blocking = [p for p in problems if "protocol repeats" not in p]
advisory = [p for p in problems if "protocol repeats" in p]

for problem in blocking:
    st.error(problem)
for problem in advisory:
    st.warning(problem)
if not problems:
    st.success("All checks passed.")

left, right = st.columns([1, 3])

with left:
    dry = st.button("Write driver only", help="Create the campaign directory and driver.sh without spawning anything.")
    go = st.button("Launch", type="primary", disabled=bool(blocking))

if dry or go:
    campaign.run_dir = None  # allocated by the launcher
    campaign.campaign_id = new_campaign_id(name or f"{engine_name}-{method.value}")
    try:
        campaign_id = launcher.launch_campaign(
            campaign, engine, cfg, dry_run=dry
        )
    except launcher.LaunchError as exc:
        st.error(f"refused: {exc}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"launch failed: {exc!r}")
    else:
        if dry:
            st.info(
                f"Wrote `{cfg.runs_root / campaign_id}/driver.sh` without launching. "
                f"Inspect or edit it, then run it by hand, or use **Resume** on the "
                f"Runs page."
            )
        else:
            st.success(f"Launched `{campaign_id}`. Watch it on the **Runs** page.")
