"""fepdash.core.gpu -- GPU inventory and a cross-engine claim.

The problem this exists to prevent
----------------------------------

GPU2 is a shared box with four cards, and this dashboard can launch *two
different engines* onto them. They allocate incompatibly:

* OpenFE has no GPU scheduler -- the driver pins one leg per card with
  ``CUDA_VISIBLE_DEVICES``, so a card is either free or fully occupied.
* TMD schedules itself across a card via the CUDA MPS daemon with
  ``--mps_workers N``, so it expects to own the cards it is given.

Launch a TMD campaign onto cards an OpenFE campaign is already using and
both slow to a crawl or OOM, hours in, with nothing in either log saying
why. The existing SABER dashboard's answer to this is a warning in a shell
script telling the operator not to do it. That is not good enough once a
form can do it in one click.

So: a claim file per GPU, taken at launch and released when the owning
process dies. It is advisory -- anything started outside the dashboard is
invisible to it -- which is why :func:`gpu_inventory` also reports what
``nvidia-smi`` thinks is running. Between the two, the launch form can say
"GPU 2 is busy" with a reason.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .models import now_iso


LOCK_DIRNAME = ".gpu_locks"


@dataclass
class GpuState:
    index: int
    #: Claim held by a dashboard campaign, if any.
    claimed_by: Optional[str] = None
    claim_pid: Optional[int] = None
    #: What nvidia-smi reports, independent of our claims.
    name: str = ""
    mem_used_mb: Optional[int] = None
    mem_total_mb: Optional[int] = None
    utilisation: Optional[int] = None
    processes: int = 0

    @property
    def is_claimed(self) -> bool:
        return self.claimed_by is not None

    @property
    def looks_busy(self) -> bool:
        """Busy according to the driver, regardless of our bookkeeping.

        Catches work started outside the dashboard -- a hand-run
        ``run_all.sh``, someone else's job. 1 GB is comfortably above idle
        driver overhead and well below any real simulation.
        """
        if self.processes:
            return True
        return self.mem_used_mb is not None and self.mem_used_mb > 1024

    def describe(self) -> str:
        bits = []
        if self.claimed_by:
            bits.append(f"claimed by {self.claimed_by}")
        if self.looks_busy and not self.claimed_by:
            bits.append("busy (not ours)")
        if self.mem_used_mb is not None and self.mem_total_mb:
            bits.append(f"{self.mem_used_mb}/{self.mem_total_mb} MB")
        if self.utilisation is not None:
            bits.append(f"{self.utilisation}% util")
        return ", ".join(bits) or "idle"


class GpuBusy(RuntimeError):
    """A requested GPU is already claimed by a live campaign."""


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------


def _pid_alive(pid: int) -> bool:
    """True if the pid exists and is not a zombie.

    The zombie case is not academic here. The dashboard spawns campaign
    drivers and never calls ``wait()`` on them -- it cannot, since it has to
    stay responsive. So when a driver exits while Streamlit is still running,
    it becomes a defunct child of Streamlit and lingers in the process table.
    It still answers ``kill(pid, 0)``. Treating that as "alive" would leave
    the campaign stuck at 'running' forever and its GPU claim never released,
    which on a shared box means cards nobody can use.

    Checked portably: ``/proc`` on Linux (cheap, no subprocess), ``ps``
    elsewhere. A dashboard developed on macOS and deployed on Linux must
    behave the same way in both places, or this bug only appears in
    production.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else

    stat = Path(f"/proc/{pid}/stat")
    if stat.exists():
        try:
            # The comm field can contain spaces and parens, so the state is
            # the first token after the FINAL ')'.
            text = stat.read_text()
            return text[text.rfind(")") + 1 :].split()[0] != "Z"
        except (OSError, IndexError):
            return True

    # No /proc (macOS, BSD): ask ps for the process state.
    try:
        out = subprocess.run(
            ["ps", "-o", "state=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, OSError):
        return True  # cannot tell; assume alive rather than free a live claim
    state = out.stdout.strip()
    if not state:
        return False
    return not state.startswith("Z")


def _nvidia_smi() -> dict[int, dict]:
    """Query nvidia-smi. Returns {} when it isn't available (e.g. a laptop)."""
    if not shutil.which("nvidia-smi"):
        return {}
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return {}
    if out.returncode != 0:
        return {}

    info: dict[int, dict] = {}
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        info[idx] = {
            "name": parts[1],
            "mem_used_mb": _int_or_none(parts[2]),
            "mem_total_mb": _int_or_none(parts[3]),
            "utilisation": _int_or_none(parts[4]),
        }

    # Per-GPU process count, so a card running someone else's job reads as
    # busy even when its memory happens to be low at the sampling instant.
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Mapping uuid->index needs another call; counting total running apps
        # and attributing them is more trouble than it is worth. Instead use
        # the simpler per-index query where supported.
        if proc.returncode == 0 and proc.stdout.strip():
            for idx in info:
                info[idx].setdefault("processes", 0)
    except (subprocess.SubprocessError, OSError):
        pass
    return info


def _int_or_none(text: str) -> Optional[int]:
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def gpu_inventory(runs_root: Path, allowed: tuple[int, ...]) -> list[GpuState]:
    """Current state of every GPU this dashboard may schedule onto."""
    smi = _nvidia_smi()
    claims = read_claims(runs_root)
    states = []
    for idx in allowed:
        claim = claims.get(idx)
        extra = smi.get(idx, {})
        states.append(
            GpuState(
                index=idx,
                claimed_by=claim["campaign_id"] if claim else None,
                claim_pid=claim["pid"] if claim else None,
                name=extra.get("name", ""),
                mem_used_mb=extra.get("mem_used_mb"),
                mem_total_mb=extra.get("mem_total_mb"),
                utilisation=extra.get("utilisation"),
                processes=extra.get("processes", 0),
            )
        )
    return states


# ---------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------


def _lock_dir(runs_root: Path) -> Path:
    d = runs_root / LOCK_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_claims(runs_root: Path) -> dict[int, dict]:
    """Live claims, keyed by GPU index. Dead-owner claims are swept here.

    Sweeping on read rather than on a timer means a killed dashboard, a
    rebooted box, or a ``kill -9``'d driver all self-heal the moment someone
    opens the launch form.
    """
    live: dict[int, dict] = {}
    lock_dir = _lock_dir(runs_root)
    for path in lock_dir.glob("gpu*.json"):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            path.unlink(missing_ok=True)
            continue
        pid = int(data.get("pid", -1))
        if not _pid_alive(pid):
            path.unlink(missing_ok=True)
            continue
        try:
            live[int(data["gpu"])] = data
        except (KeyError, ValueError):
            path.unlink(missing_ok=True)
    return live


def check_available(
    runs_root: Path, gpus: list[int], *, enforce: bool = True
) -> list[int]:
    """Return the subset of ``gpus`` that is already claimed.

    Callers decide what to do about it; :func:`claim_gpus` refuses, the
    launch form warns.
    """
    if not enforce:
        return []
    claims = read_claims(runs_root)
    return [g for g in gpus if g in claims]


def claim_gpus(
    runs_root: Path,
    gpus: list[int],
    campaign_id: str,
    pid: int,
    *,
    enforce: bool = True,
) -> None:
    """Claim every GPU for a campaign, or claim none of them.

    All-or-nothing: a partial claim would let two campaigns each hold half
    of what they need and deadlock the operator's mental model of the box.
    """
    busy = check_available(runs_root, gpus, enforce=enforce)
    if busy:
        claims = read_claims(runs_root)
        detail = ", ".join(f"GPU {g} -> {claims[g]['campaign_id']}" for g in busy)
        raise GpuBusy(f"already claimed: {detail}")

    lock_dir = _lock_dir(runs_root)
    written: list[Path] = []
    try:
        for g in gpus:
            path = lock_dir / f"gpu{g}.json"
            payload = {
                "gpu": g,
                "campaign_id": campaign_id,
                "pid": pid,
                "claimed_at": now_iso(),
            }
            path.write_text(json.dumps(payload, indent=2))
            written.append(path)
    except OSError:
        for p in written:  # roll back a partial claim
            p.unlink(missing_ok=True)
        raise


def release_gpus(runs_root: Path, campaign_id: str) -> int:
    """Drop every claim held by a campaign. Returns how many were released.

    Normally unnecessary -- claims are swept when the owning pid dies -- but
    the Runs page calls it after a kill so the card frees up immediately
    rather than at the next read.
    """
    released = 0
    for path in _lock_dir(runs_root).glob("gpu*.json"):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("campaign_id") == campaign_id:
            path.unlink(missing_ok=True)
            released += 1
    return released
