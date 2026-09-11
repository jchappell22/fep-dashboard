"""fepdash.core.models -- the domain objects shared by every page.

One vocabulary for two engines
------------------------------

OpenFE and TMD look nothing alike at the command line, but they compute the
same thing from the same inputs:

    ligands.sdf + protein.pdb  ->  edge network  ->  per-leg GPU runs
                               ->  dDDG per edge + dG per ligand

So the dashboard models a **campaign** (one protein, one ligand set, one
engine, one method) containing **legs** (the unit of GPU work), and each
engine adapter maps that onto its own CLI. Nothing above the adapter layer
knows what an ``openfe quickrun`` or an ``--mps_workers`` is.

Where state lives
-----------------

Two places, deliberately:

``state.db`` (sqlite, at ``runs_root``)
    Only what the filesystem cannot tell us: the pid, the operator's
    intent, and the exit code. One row per campaign.

The campaign directory
    Everything else. Leg status is *derived* by scanning the tree on every
    read, never cached in the DB. A leg that finished while the dashboard
    was down still shows as done, and deleting a campaign directory cannot
    leave the DB describing runs that no longer exist.

Campaign directory layout (identical for both engines)::

    runs/<campaign_id>/
      campaign.json      the manifest -- engine, method, inputs, params, GPUs
      driver.sh          the exact script that was spawned (see driver.py)
      stdout.log         driver stdout; the Runs page tails this
      stderr.log         driver stderr
      plan/              engine planning output (transformations/ or map.json)
      legs/<leg>/        per-leg status.json + log, dashboard-owned
      work/<leg>/        engine scratch (openfe -d)
      results/           engine result artifacts (openfe quickrun -o)
      gathered/          ddg.tsv / dg.tsv -- whatever the tables page reads
      .stage             the driver's current stage name, one line
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Method(str, enum.Enum):
    """What is being calculated.

    Not every engine supports every one -- each engine TOML declares its own
    ``methods`` list and the Launch page only offers those. ``EDGE`` exists
    because TMD's launcher has a first-class single-edge subcommand, which is
    how you re-run one bad edge out of a finished network without redoing the
    whole thing.
    """

    RBFE = "rbfe"
    ABFE = "abfe"
    EDGE = "edge"

    @property
    def label(self) -> str:
        return {
            "rbfe": "RBFE (relative, whole network)",
            "abfe": "ABFE (absolute)",
            "edge": "Single edge (A → B)",
        }[self.value]


class CampaignStatus(str, enum.Enum):
    """Lifecycle of the driver process.

    QUEUED exists for the same reason it does in saber-dashboard: the DB row
    is written *before* ``Popen``, so a spawn failure leaves a forensic trail
    rather than a missing campaign.
    """

    QUEUED = "queued"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"
    KILLED = "killed"

    @property
    def is_terminal(self) -> bool:
        return self in (
            CampaignStatus.FINISHED,
            CampaignStatus.FAILED,
            CampaignStatus.KILLED,
        )


class LegStatus(str, enum.Enum):
    """State of one unit of GPU work, derived from the filesystem."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class Stage(str, enum.Enum):
    """Driver stages, written one-per-line to ``.stage``.

    Planning is a separate stage because for OpenFE it generates am1bcc
    partial charges for every ligand -- minutes to hours of CPU before a
    single GPU is touched. A campaign sitting in PLAN is not stuck.
    """

    PLAN = "plan"
    RUN = "run"
    GATHER = "gather"
    DONE = "done"


# ---------------------------------------------------------------------------
# Campaign
# ---------------------------------------------------------------------------


@dataclass
class Campaign:
    """One protein + one ligand set + one engine + one method.

    ``params`` holds the engine-specific knobs straight from the launch form
    (``local_md_steps``, ``n_repeats``, ``mps_workers``, ...). It is passed
    to the adapter as render variables and is deliberately untyped here --
    adding an engine must not mean editing this class.
    """

    campaign_id: str
    name: str
    engine: str
    method: Method

    protein: Path
    ligands: Path
    cofactors: Optional[Path] = None
    settings_yaml: Optional[Path] = None

    gpus: list[int] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    run_dir: Optional[Path] = None
    created_at: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = now_iso()
        if isinstance(self.method, str):
            self.method = Method(self.method)
        for attr in ("protein", "ligands", "cofactors", "settings_yaml", "run_dir"):
            val = getattr(self, attr)
            if isinstance(val, str):
                setattr(self, attr, Path(val))

    # -- directory accessors; the single source of truth for the layout ----

    @property
    def plan_dir(self) -> Path:
        return self._root / "plan"

    @property
    def legs_dir(self) -> Path:
        return self._root / "legs"

    @property
    def work_dir(self) -> Path:
        return self._root / "work"

    @property
    def results_dir(self) -> Path:
        return self._root / "results"

    @property
    def gathered_dir(self) -> Path:
        return self._root / "gathered"

    @property
    def stage_file(self) -> Path:
        return self._root / ".stage"

    @property
    def manifest_path(self) -> Path:
        return self._root / "campaign.json"

    @property
    def driver_path(self) -> Path:
        return self._root / "driver.sh"

    @property
    def _root(self) -> Path:
        if self.run_dir is None:
            raise ValueError(
                f"campaign {self.campaign_id} has no run_dir; it has not been "
                "allocated on disk yet"
            )
        return self.run_dir

    # -- serialisation ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["method"] = self.method.value
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Campaign":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        # Tolerate manifests written by a newer version of the dashboard:
        # drop unknown keys rather than blowing up the Runs page.
        return cls(**{k: v for k, v in d.items() if k in known})

    def current_stage(self) -> Optional[Stage]:
        """Read ``.stage``. None if the driver has not started one yet."""
        try:
            raw = self.stage_file.read_text().strip()
        except (OSError, ValueError):
            return None
        try:
            return Stage(raw)
        except ValueError:
            return None


@dataclass
class Leg:
    """One unit of GPU work within a campaign.

    For OpenFE this is one ``openfe quickrun`` of one transformation JSON --
    one edge-leg such as ``rbfe_ligA_complex_ligB_complex``. For TMD the
    whole graph is a single process, so a TMD campaign has exactly one leg
    whose progress comes from parsing the driver log instead.
    """

    leg_id: str
    status: LegStatus
    gpu: Optional[int] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_s: Optional[float] = None
    log_path: Optional[Path] = None
    error: str = ""

    @property
    def edge(self) -> str:
        """Best-effort edge name, for grouping complex/solvent side by side.

        OpenFE names legs ``rbfe_<A>_<leg>_<B>_<leg>``; strip the leg type so
        the two halves of an edge land on one row. Anything unrecognised is
        returned unchanged rather than mangled.
        """
        for suffix in ("_complex", "_solvent", "_vacuum"):
            if self.leg_id.endswith(suffix):
                return self.leg_id[: -len(suffix)].replace(suffix, "")
        return self.leg_id

    @property
    def leg_type(self) -> str:
        for suffix in ("complex", "solvent", "vacuum"):
            if self.leg_id.endswith("_" + suffix):
                return suffix
        return ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def now_iso() -> str:
    """ISO-8601 UTC with a 'Z' suffix. Matches saber-dashboard's format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_campaign_id(name: str) -> str:
    """``20260911-143005_tyk2-rbfe`` -- sortable, greppable, filesystem-safe."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in name.lower())
    slug = "-".join(filter(None, slug.split("-")))[:40]
    return f"{stamp}_{slug}" if slug else stamp
