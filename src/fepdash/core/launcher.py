"""fepdash.core.launcher -- spawn and kill campaign drivers.

Ordering matters here, and the order is the same one saber-dashboard uses
for the same reasons:

1. allocate the campaign directory
2. write ``campaign.json`` and ``driver.sh``    <- forensic trail first
3. INSERT the row as ``queued``                 <- visible before spawning
4. ``Popen(..., start_new_session=True)``
5. claim the GPUs against the new pid
6. UPDATE the row to ``running`` with the pid

If step 4 raises (binary missing, permission denied, out of memory) the row
already exists and is flipped to ``failed`` with the reason in ``notes``,
rather than a campaign that silently never appears.

``start_new_session=True`` is what makes the run survive the dashboard:
the driver becomes its own process-group leader, so Streamlit restarting,
crashing, or being Ctrl-C'd cannot take a 12-hour campaign down with it. It
is also what makes :func:`kill_campaign` work -- ``os.killpg`` on the group
takes down the driver *and* every ``openfe quickrun`` it spawned, which a
plain ``proc.kill()`` would orphan onto the GPUs.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional

from . import db as _db
from . import gpu as _gpu
from .config import Config
from .driver import write_driver
from .engines.base import Engine
from .models import Campaign, CampaignStatus, new_campaign_id


class LaunchError(RuntimeError):
    """Refused before anything was spawned."""


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------


def preflight(campaign: Campaign, engine: Engine, cfg: Config) -> list[str]:
    """Problems that would waste GPU hours. Empty list means go.

    Cheap checks only -- everything here is a stat or a comparison. The
    expensive truth (does this protein PDB actually parse in OpenMM) can
    only be learned by running, which is what the plan stage is for.
    """
    problems: list[str] = []

    if not engine.supports(campaign.method):
        problems.append(
            f"engine '{engine.name}' does not declare method "
            f"'{campaign.method.value}' (methods: {', '.join(engine.methods) or 'none'})"
        )

    for label, path in (("protein", campaign.protein), ("ligands", campaign.ligands)):
        # Distinguish "you haven't chosen one" from "the path is wrong" --
        # the empty Path() renders as "." and reads like a real broken path.
        if not path or str(path) in ("", "."):
            problems.append(f"no {label} file selected")
        elif not Path(path).is_file():
            problems.append(f"{label} file not found: {path}")

    for label, path in (
        ("cofactors", campaign.cofactors),
        ("settings yaml", campaign.settings_yaml),
    ):
        if path and not Path(path).is_file():
            problems.append(f"{label} file not found: {path}")

    # GPU checks only apply to engines the dashboard schedules. TMD's
    # launcher picks the most-free card itself, so there is nothing for us
    # to select, validate, or claim -- and pretending otherwise would put a
    # claim on a card the engine may well not use.
    if not engine.picks_own_gpu:
        if not campaign.gpus:
            problems.append("no GPUs selected")
        bad = [g for g in campaign.gpus if g not in cfg.allowed_gpus]
        if bad:
            problems.append(
                f"GPU(s) {bad} are not in allowed_gpus {list(cfg.allowed_gpus)} "
                f"(see [gpu] allowed in config.toml)"
            )

        busy = _gpu.check_available(
            cfg.runs_root, campaign.gpus, enforce=cfg.enforce_gpu_locks
        )
        if busy:
            claims = _gpu.read_claims(cfg.runs_root)
            detail = ", ".join(
                f"GPU {g} held by {claims[g]['campaign_id']}" for g in busy if g in claims
            )
            problems.append(f"GPU already claimed: {detail}")

    # OpenFE's `gather --report dg` builds an MLE across repeats and needs at
    # least two per edge. Better to say so on the form than 18 hours later.
    min_repeats = engine.stage(campaign.method, "gather").get("min_repeats_for_dg", 1)
    n_repeats = int(campaign.params.get("n_repeats", 1) or 1)
    if n_repeats < int(min_repeats):
        problems.append(
            f"{engine.name} needs >= {min_repeats} protocol repeats per edge to "
            f"report per-ligand dG, but n_repeats={n_repeats}. Per-edge ddG will "
            f"still work."
        )

    return problems


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------


def allocate_run_dir(runs_root: Path, campaign_id: str) -> Path:
    """Create ``runs/<campaign_id>/``, refusing to reuse an existing one."""
    run_dir = runs_root / campaign_id
    if run_dir.exists():
        raise LaunchError(f"campaign directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    for sub in ("plan", "legs", "work", "results", "gathered"):
        (run_dir / sub).mkdir()
    return run_dir


def launch_campaign(
    campaign: Campaign,
    engine: Engine,
    cfg: Config,
    *,
    python: str = "python",
    dry_run: bool = False,
) -> str:
    """Spawn a campaign. Returns its id.

    ``dry_run`` writes the directory, manifest and driver script but does
    not spawn -- the Launch page uses it so an operator can read the exact
    script before committing a card for half a day.
    """
    problems = preflight(campaign, engine, cfg)
    # A repeats warning is advisory; a missing file is not. Only the latter
    # blocks, so a deliberate single-repeat ddG-only run stays possible.
    blocking = [p for p in problems if "protocol repeats" not in p]
    if blocking:
        raise LaunchError("; ".join(blocking))

    if not campaign.campaign_id:
        campaign.campaign_id = new_campaign_id(campaign.name)

    _db.init_db(cfg.db_path)
    run_dir = allocate_run_dir(cfg.runs_root, campaign.campaign_id)
    campaign.run_dir = run_dir

    # 2. Forensic trail before anything can fail.
    campaign.manifest_path.write_text(json.dumps(campaign.to_dict(), indent=2))
    driver_path = write_driver(
        campaign, engine, python=python, scripts_dir=cfg.scripts_dir
    )

    if dry_run:
        return campaign.campaign_id

    # 3. Row first, so a spawn failure is visible in the UI.
    _db.insert_campaign(campaign, cfg.db_path)

    stdout_log = run_dir / "stdout.log"
    stderr_log = run_dir / "stderr.log"
    # Unbuffered append so `tail -f` works while the campaign is in flight.
    stdout_fh = stdout_log.open("ab", buffering=0)
    stderr_fh = stderr_log.open("ab", buffering=0)

    env = dict(os.environ)
    env["FEPDASH_CAMPAIGN"] = campaign.campaign_id
    env["GPUS"] = " ".join(str(g) for g in campaign.gpus)

    try:
        proc = subprocess.Popen(
            ["bash", str(driver_path)],
            stdin=subprocess.DEVNULL,
            stdout=stdout_fh,
            stderr=stderr_fh,
            env=env,
            cwd=str(run_dir),
            start_new_session=True,
            close_fds=True,
        )
    except Exception as exc:
        stdout_fh.close()
        stderr_fh.close()
        _db.mark_terminal(
            campaign.campaign_id,
            CampaignStatus.FAILED,
            cfg.db_path,
            notes=f"failed to spawn driver: {exc!r}",
        )
        raise
    finally:
        # The kernel keeps the child's inherited fds open; ours would leak.
        stdout_fh.close()
        stderr_fh.close()

    # 5. Claim the cards against the live pid. If this fails we have a
    #    running driver with no claim -- kill it rather than leave an
    #    unaccounted-for job on a shared box.
    #
    #    Skipped for engines that pick their own GPU: we cannot know which
    #    card tmd-submit chose, and a claim on a card it isn't using would
    #    block other campaigns for no reason.
    try:
        if not engine.picks_own_gpu:
            _gpu.claim_gpus(
                cfg.runs_root,
                campaign.gpus,
                campaign.campaign_id,
                proc.pid,
                enforce=cfg.enforce_gpu_locks,
            )
    except _gpu.GpuBusy as exc:
        _kill_group(proc.pid)
        _db.mark_terminal(
            campaign.campaign_id,
            CampaignStatus.FAILED,
            cfg.db_path,
            notes=f"GPU claim lost in a race, driver killed: {exc}",
        )
        raise LaunchError(str(exc)) from exc

    _db.mark_running(campaign.campaign_id, proc.pid, cfg.db_path)
    return campaign.campaign_id


# ---------------------------------------------------------------------------
# Kill
# ---------------------------------------------------------------------------


def _kill_group(pid: int, *, grace_s: float = 10.0) -> str:
    """SIGTERM the process group, then SIGKILL what's left.

    The group, not the pid: the driver's children are the actual engine
    processes holding the GPUs. Killing only the driver would orphan them
    and leave the cards occupied with nothing tracking them.
    """
    if pid <= 0:
        return "no pid recorded"
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return "already gone"
    except PermissionError:
        return "permission denied (started by another user?)"

    deadline = time.time() + grace_s
    while time.time() < deadline:
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            return "terminated"
        time.sleep(0.25)

    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return "terminated"
    return "killed (SIGKILL after grace period)"


def kill_campaign(campaign_id: str, cfg: Config) -> str:
    """Stop a running campaign and free its cards."""
    row = _db.get_campaign(campaign_id, cfg.db_path)
    if row is None:
        return f"no such campaign: {campaign_id}"
    status = CampaignStatus(row["status"])
    if status.is_terminal:
        return f"campaign is already {status.value}"

    pid = row.get("pid")
    if not pid:
        # Queued but never spawned -- nothing to signal.
        _db.mark_terminal(
            campaign_id, CampaignStatus.KILLED, cfg.db_path,
            notes="killed before the driver was spawned",
        )
        _gpu.release_gpus(cfg.runs_root, campaign_id)
        return "cancelled before launch"

    result = _kill_group(int(pid))
    _db.mark_terminal(
        campaign_id, CampaignStatus.KILLED, cfg.db_path, notes=f"operator kill: {result}"
    )
    _gpu.release_gpus(cfg.runs_root, campaign_id)
    return result


def resume_campaign(campaign_id: str, cfg: Config) -> str:
    """Re-spawn a stopped campaign's existing driver script.

    The driver is written to be resumable -- completed legs are skipped -- so
    recovering from a crash, a kill, or a box reboot is re-running the same
    file. Nothing is regenerated, so a hand-edited driver.sh is respected.
    """
    row = _db.get_campaign(campaign_id, cfg.db_path)
    if row is None:
        raise LaunchError(f"no such campaign: {campaign_id}")
    if not CampaignStatus(row["status"]).is_terminal:
        raise LaunchError("campaign is still active; kill it before resuming")

    run_dir = Path(row["run_dir"])
    driver_path = run_dir / "driver.sh"
    if not driver_path.is_file():
        raise LaunchError(f"no driver script at {driver_path}")

    gpus = row.get("gpus") or []
    busy = _gpu.check_available(cfg.runs_root, gpus, enforce=cfg.enforce_gpu_locks)
    if busy:
        raise LaunchError(f"GPU(s) {busy} are claimed by another campaign")

    stdout_fh = (run_dir / "stdout.log").open("ab", buffering=0)
    stderr_fh = (run_dir / "stderr.log").open("ab", buffering=0)
    env = dict(os.environ)
    env["GPUS"] = " ".join(str(g) for g in gpus)
    try:
        proc = subprocess.Popen(
            ["bash", str(driver_path)],
            stdin=subprocess.DEVNULL,
            stdout=stdout_fh,
            stderr=stderr_fh,
            env=env,
            cwd=str(run_dir),
            start_new_session=True,
            close_fds=True,
        )
    finally:
        stdout_fh.close()
        stderr_fh.close()

    _gpu.claim_gpus(
        cfg.runs_root, gpus, campaign_id, proc.pid, enforce=cfg.enforce_gpu_locks
    )
    _db.mark_running(campaign_id, proc.pid, cfg.db_path)
    return f"resumed as pid {proc.pid}"
