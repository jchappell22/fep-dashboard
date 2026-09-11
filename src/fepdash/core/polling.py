"""fepdash.core.polling -- notice that a driver exited; write the transition.

The launcher returns the moment the driver is spawned and never waits. The
running -> finished/failed transition is therefore written here, by whoever
loads a page next.

This is a pull model on purpose. A background thread in Streamlit dies with
the session, a daemon would be another thing to keep alive on GPU2, and the
transition is only ever *observed* -- not acted on -- so learning about it
one page-load late costs nothing.

Why the driver is not simply trusted to record its own exit
-----------------------------------------------------------

It cannot record the ways it dies that matter most: ``SIGKILL``, an OOM
kill, or the box rebooting. So liveness is established from outside, by
checking the pid, and the driver's own ``.stage`` file is used only to tell
a clean finish from an abrupt one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import db as _db
from . import gpu as _gpu
from .config import Config
from .models import CampaignStatus, Stage


def _pid_alive(pid: int) -> bool:
    """Shared with the GPU claim sweeper -- one definition of 'alive'."""
    return _gpu._pid_alive(pid)


def _classify(row: dict[str, Any]) -> tuple[CampaignStatus, str]:
    """Decide how a no-longer-running campaign ended.

    The driver writes ``done`` into ``.stage`` as its last act, and calls
    ``fail`` (which also writes ``done``) on an error. So the marker alone
    cannot separate success from failure -- the campaign directory has to be
    consulted for whether any leg failed.
    """
    run_dir = Path(row["run_dir"])
    stage_file = run_dir / ".stage"

    try:
        stage_raw = stage_file.read_text().strip()
    except OSError:
        return (
            CampaignStatus.FAILED,
            "driver exited without ever writing a stage marker -- it probably "
            "could not start (check stderr.log)",
        )

    if stage_raw != Stage.DONE.value:
        return (
            CampaignStatus.FAILED,
            f"driver died during stage '{stage_raw}' without finishing "
            f"(killed, OOM, or the box went down)",
        )

    failed_legs = _count_failed_legs(run_dir)
    if failed_legs:
        return (
            CampaignStatus.FINISHED,
            f"completed with {failed_legs} failed leg(s) -- re-run to retry them",
        )
    return CampaignStatus.FINISHED, ""


def _count_failed_legs(run_dir: Path) -> int:
    legs_dir = run_dir / "legs"
    if not legs_dir.is_dir():
        return 0
    count = 0
    for status_path in legs_dir.glob("*/status.json"):
        try:
            if '"status": "failed"' in status_path.read_text():
                count += 1
        except OSError:
            continue
    return count


def poll_active(cfg: Config) -> dict[str, str]:
    """Check every queued/running campaign; write any terminal transitions.

    Returns ``{campaign_id: new_status}`` for whatever changed, so a page
    can report "2 campaigns finished while you were away".
    """
    changed: dict[str, str] = {}

    for row in _db.active_campaigns(cfg.db_path):
        pid = row.get("pid")
        campaign_id = row["campaign_id"]

        if not pid:
            # Queued with no pid: the launcher was interrupted between the
            # INSERT and the Popen. Nothing is running, and nothing ever
            # will be.
            if row["status"] == CampaignStatus.QUEUED.value:
                _db.mark_terminal(
                    campaign_id,
                    CampaignStatus.FAILED,
                    cfg.db_path,
                    notes="queued but never spawned (dashboard interrupted mid-launch)",
                )
                changed[campaign_id] = CampaignStatus.FAILED.value
            continue

        if _pid_alive(int(pid)):
            continue

        status, notes = _classify(row)
        _db.mark_terminal(campaign_id, status, cfg.db_path, notes=notes)
        # Free the cards now rather than at the next claim sweep, so the
        # launch form is immediately correct.
        _gpu.release_gpus(cfg.runs_root, campaign_id)
        changed[campaign_id] = status.value

    return changed
