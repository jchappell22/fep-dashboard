"""fepdash.core.state -- derive live progress by reading the campaign tree.

Nothing in here is cached in the database. Leg state is recomputed from the
filesystem on every call, which is what lets the dashboard be restarted,
moved, or run twice at once without ever disagreeing with the disk about
what finished.

Cost: one ``glob`` plus a small JSON read per leg. For a 40-edge RBFE
network that is ~80 stats, which is fine at Streamlit's rerun rate. The
expensive thing -- opening a live NetCDF to read the sampler iteration --
is deliberately *not* done here; see :func:`openmm_iteration` for why.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from .models import Campaign, Leg, LegStatus, Stage


@dataclass
class Progress:
    """Everything the Runs page needs about one campaign's legs."""

    legs: list[Leg]
    stage: Optional[Stage]

    @property
    def done(self) -> int:
        return sum(1 for l in self.legs if l.status is LegStatus.DONE)

    @property
    def failed(self) -> int:
        return sum(1 for l in self.legs if l.status is LegStatus.FAILED)

    @property
    def running(self) -> int:
        return sum(1 for l in self.legs if l.status is LegStatus.RUNNING)

    @property
    def pending(self) -> int:
        return sum(1 for l in self.legs if l.status is LegStatus.PENDING)

    @property
    def total(self) -> int:
        return len(self.legs)

    @property
    def fraction(self) -> float:
        return (self.done / self.total) if self.total else 0.0

    def mean_leg_seconds(self) -> Optional[float]:
        durations = [l.duration_s for l in self.legs if l.duration_s]
        return sum(durations) / len(durations) if durations else None

    def eta(self, n_gpus: int) -> Optional[timedelta]:
        """Remaining wall-clock, from the mean of completed legs.

        Returns None until at least one leg has finished -- an ETA computed
        from nothing is worse than no ETA, because people plan around it.
        Deliberately crude: it assumes legs are interchangeable, which they
        are not (complex legs run several times longer than solvent ones),
        so treat it as an order of magnitude.
        """
        mean = self.mean_leg_seconds()
        if mean is None:
            return None
        remaining = self.pending + self.running
        if remaining <= 0:
            return timedelta(0)
        return timedelta(seconds=remaining * mean / max(n_gpus, 1))


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------


def scan_legs(campaign: Campaign, *, expected: Optional[list[str]] = None) -> list[Leg]:
    """Read every ``legs/<leg>/status.json`` the driver has written.

    ``expected`` -- the leg names planning produced -- lets legs that have
    not started yet appear as PENDING. Without it a campaign in its first
    minutes looks empty rather than queued, which reads as "broken".
    """
    legs: dict[str, Leg] = {}

    legs_dir = campaign.legs_dir
    if legs_dir.is_dir():
        for status_path in sorted(legs_dir.glob("*/status.json")):
            leg = _read_leg(status_path)
            if leg is not None:
                legs[leg.leg_id] = leg

    for name in expected or []:
        if name not in legs:
            legs[name] = Leg(leg_id=name, status=LegStatus.PENDING)

    return sorted(legs.values(), key=lambda l: l.leg_id)


def _read_leg(status_path: Path) -> Optional[Leg]:
    try:
        data = json.loads(status_path.read_text())
    except (OSError, json.JSONDecodeError):
        # The driver writes status.json atomically via a temp file + mv, so
        # a torn read should be impossible -- but a truncated file from a
        # full disk or a hard reset should not take the page down.
        return None

    leg_id = data.get("leg_id") or status_path.parent.name
    try:
        status = LegStatus(data.get("status", "pending"))
    except ValueError:
        status = LegStatus.PENDING

    gpu = data.get("gpu")
    duration = data.get("duration_s")
    log_path = status_path.parent / "leg.log"

    return Leg(
        leg_id=leg_id,
        status=status,
        gpu=_maybe_int(gpu),
        started_at=data.get("started_at"),
        finished_at=data.get("finished_at"),
        duration_s=_maybe_float(duration),
        log_path=log_path if log_path.is_file() else None,
        error=_tail_error(log_path) if status is LegStatus.FAILED else "",
    )


def _maybe_int(value) -> Optional[int]:
    try:
        return int(str(value).split(",")[0])  # "0,1,2" -> 0 for the single-proc case
    except (TypeError, ValueError):
        return None


def _maybe_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def expected_legs(campaign: Campaign, jobs_glob: str) -> list[str]:
    """Leg names implied by the planning output.

    For OpenFE these are the transformation JSON stems, e.g.
    ``rbfe_ligA_complex_ligB_complex``. For a single-process engine the glob
    is empty and the campaign has exactly one leg.
    """
    if not jobs_glob:
        return ["graph"]
    plan_dir = campaign.plan_dir
    if not plan_dir.is_dir():
        return []
    return sorted(p.stem for p in plan_dir.glob(jobs_glob))


def progress(campaign: Campaign, jobs_glob: str) -> Progress:
    names = expected_legs(campaign, jobs_glob)
    return Progress(legs=scan_legs(campaign, expected=names), stage=campaign.current_stage())


# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------


def tail(path: Path, n: int = 200, *, max_bytes: int = 256_000) -> str:
    """Last ``n`` lines, reading only the tail of the file.

    Leg logs from a long OpenMM run reach hundreds of MB. Reading one whole
    to show the last screenful would stall the page and churn memory on
    every Streamlit rerun, so seek from the end instead.
    """
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > max_bytes:
                fh.seek(size - max_bytes)
                fh.readline()  # discard the partial first line
            data = fh.read()
    except OSError:
        return ""
    lines = data.decode("utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])


_ERROR_HINTS = re.compile(
    r"(Error|Exception|Traceback|CUDA|out of memory|Killed|assert)", re.IGNORECASE
)


def _tail_error(log_path: Path, lines: int = 40) -> str:
    """A one-line hint at why a leg failed, for the Runs table.

    Scans only the tail; the full log is one click away, so this needs to be
    suggestive, not complete.
    """
    text = tail(log_path, lines)
    if not text:
        return ""
    for line in reversed(text.splitlines()):
        if _ERROR_HINTS.search(line):
            return line.strip()[:200]
    return text.splitlines()[-1].strip()[:200] if text.splitlines() else ""


# ---------------------------------------------------------------------------
# Deliberately not done here
# ---------------------------------------------------------------------------


def openmm_iteration(nc_path: Path) -> Optional[int]:
    """Current sampler iteration from a live ``simulation.nc``.

    NOT called during a normal page render. Opening a multistate reporter
    imports ``openmmtools`` and takes a handle on a NetCDF file that a
    running simulation is actively writing; doing that for every leg on
    every Streamlit rerun would make the page crawl and put read handles on
    live simulations.

    It is exposed for the "inspect one leg" view, where the operator has
    asked for this specific leg and can wait a second for it. Callers should
    wrap it in ``st.cache_data(ttl=...)`` keyed on path + mtime.
    """
    try:
        from openmmtools.multistate import MultiStateReporter
    except ImportError:
        return None
    try:
        reporter = MultiStateReporter(str(nc_path), open_mode="r")
    except Exception:
        return None
    try:
        current = reporter.read_last_iteration()
        return int(current) if current is not None else None
    except Exception:
        return None
    finally:
        try:
            reporter.close()
        except Exception:
            pass


def find_simulation_ncs(campaign: Campaign, leg_id: str) -> list[Path]:
    """Live NetCDF files under one leg's work directory, newest write first."""
    work = campaign.work_dir / leg_id
    if not work.is_dir():
        return []
    ncs = list(work.glob("**/simulation.nc"))
    return sorted(ncs, key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)


def age_seconds(path: Path) -> Optional[float]:
    """Seconds since ``path`` was last written; a stalled leg shows up here."""
    try:
        return (
            datetime.now(timezone.utc)
            - datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        ).total_seconds()
    except OSError:
        return None
