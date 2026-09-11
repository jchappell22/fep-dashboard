"""fepdash.core.fixtures -- synthetic campaigns for validating the UI.

Why this ships in the product rather than in tests/
---------------------------------------------------

Every state-parsing path in this dashboard was written against directory
layouts that could not be exercised where it was written, and the real
feedback loop is a 6-12 hour GPU run. Without a way to fabricate a campaign
tree, the first time anyone finds out the Runs page mis-reads a failed leg
is a day into a real campaign.

``python -m fepdash.core.fixtures --runs-root ./runs`` builds a campaign
directory covering the states that actually break things:

* legs done, running, pending, and failed -- all four at once
* a failed leg with a plausible CUDA traceback in its log
* result tables in both engines' formats
* an edge network with a deliberate cycle-closure outlier
* a campaign whose driver pid is dead, to exercise the poller's
  "died without finishing" classification

It writes only into ``runs_root`` and never touches a GPU.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Optional

from . import db as _db
from .models import Campaign, CampaignStatus, Method, Stage, now_iso


_LIGANDS = [
    "lig_ejm_31", "lig_ejm_42", "lig_ejm_43", "lig_ejm_46",
    "lig_ejm_47", "lig_ejm_48", "lig_ejm_50", "lig_jmc_23",
]

_CUDA_TRACEBACK = """\
Traceback (most recent call last):
  File "/home/jacob/miniconda3/envs/openfe_1-11/lib/python3.11/site-packages/openfe/protocols/openmm_rfe/equil_rfe_methods.py", line 812, in run
    sampler.minimize()
  File "/home/jacob/miniconda3/envs/openfe_1-11/lib/python3.11/site-packages/openmmtools/multistate/multistatesampler.py", line 641, in minimize
    context.applyConstraints(1e-6)
openmm.OpenMMException: Particle coordinate is NaN.  For more information, see
https://github.com/openmm/openmm/wiki/Frequently-Asked-Questions#nan
"""


def _write_leg(legs_dir: Path, name: str, status: str, **extra) -> None:
    d = legs_dir / name
    d.mkdir(parents=True, exist_ok=True)
    payload = {"leg_id": name, "status": status}
    payload.update({k: str(v) for k, v in extra.items()})
    (d / "status.json").write_text(json.dumps(payload) + "\n")

    if status == "failed":
        (d / "leg.log").write_text(
            f"starting {name}\nequilibrating...\n{_CUDA_TRACEBACK}"
        )
    elif status in ("done", "running"):
        (d / "leg.log").write_text(
            f"starting {name}\nequilibrating...\n"
            + "\n".join(f"iteration {i}/1000  dE = {random.uniform(-2, 2):+.3f}"
                        for i in range(0, 400, 50))
        )


def make_campaign(
    runs_root: Path,
    *,
    engine: str = "openfe",
    method: Method = Method.RBFE,
    name: str = "fixture-tyk2",
    campaign_id: str = "",
    status: CampaignStatus = CampaignStatus.RUNNING,
    seed: int = 7,
    single_process: Optional[bool] = None,
) -> Campaign:
    """Build one synthetic campaign directory and its DB row.

    ``single_process`` defaults to matching the real engine: TMD runs the
    whole graph in one process, OpenFE fans out per edge.
    """
    random.seed(seed)
    if single_process is None:
        single_process = engine == "tmd"
    campaign_id = campaign_id or f"fixture_{engine}_{method.value}"
    run_dir = runs_root / campaign_id
    for sub in ("plan/transformations", "legs", "work", "results", "gathered"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    campaign = Campaign(
        campaign_id=campaign_id,
        name=name,
        engine=engine,
        method=method,
        protein=runs_root / "fixture_inputs" / "protein.pdb",
        ligands=runs_root / "fixture_inputs" / "ligands.sdf",
        gpus=[0, 1],
        params={"n_repeats": 3, "n_cores": 8},
        run_dir=run_dir,
        notes="synthetic fixture -- no simulation was run",
    )

    # Inputs referenced by the manifest, so the Runs page's path display is
    # not full of broken links.
    inputs = runs_root / "fixture_inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    (inputs / "protein.pdb").write_text("REMARK fixture\nEND\n")
    (inputs / "ligands.sdf").write_text("$$$$\n")
    campaign.manifest_path.write_text(json.dumps(campaign.to_dict(), indent=2))
    campaign.driver_path.write_text("#!/usr/bin/env bash\n# fixture -- not runnable\n")

    edges = [(_LIGANDS[0], l) for l in _LIGANDS[1:]]
    edges += [(_LIGANDS[1], _LIGANDS[2]), (_LIGANDS[2], _LIGANDS[3])]

    # The two engines produce structurally different trees, and the fixture
    # has to reproduce that difference or it tests a shape that never occurs.
    #
    # A single-process engine (TMD) runs the whole graph in ONE process, so
    # its campaign has exactly one leg named `graph` and no per-edge
    # transformation files. Fabricating OpenFE-style per-edge legs for it
    # would exercise a state the real driver never writes, while leaving the
    # state it does write untested.
    if single_process:
        _write_leg(
            campaign.legs_dir, "graph",
            "running" if status is CampaignStatus.RUNNING else "done",
            gpu="auto", started_at=now_iso(),
            **({} if status is CampaignStatus.RUNNING
               else {"finished_at": now_iso(), "duration_s": 41234}),
        )
    else:
        # -- planning output: one transformation per edge, both legs ------
        leg_names = []
        for a, b in edges:
            for leg in ("complex", "solvent"):
                stem = f"rbfe_{a}_{leg}_{b}_{leg}"
                (campaign.plan_dir / "transformations" / f"{stem}.json").write_text("{}\n")
                leg_names.append(stem)

        # -- leg states: all four at once, deliberately ------------------
        for i, stem in enumerate(leg_names):
            bucket = i % 5
            if bucket in (0, 1):
                _write_leg(
                    campaign.legs_dir, stem, "done",
                    gpu=i % 2, started_at=now_iso(), finished_at=now_iso(),
                    duration_s=random.randint(3000, 20000),
                )
                (campaign.results_dir / f"{stem}.json").write_text("{}\n")
            elif bucket == 2:
                _write_leg(campaign.legs_dir, stem, "running", gpu=i % 2,
                           started_at=now_iso())
                nc = campaign.work_dir / stem / "simulation.nc"
                nc.parent.mkdir(parents=True, exist_ok=True)
                nc.write_bytes(b"CDF\x01")  # not real NetCDF; exercises the path
            elif bucket == 3:
                _write_leg(
                    campaign.legs_dir, stem, "failed", gpu=i % 2,
                    started_at=now_iso(), finished_at=now_iso(), duration_s=412,
                )
            # bucket 4 -> left absent, so it renders as PENDING

    campaign.stage_file.write_text(
        Stage.RUN.value if status is CampaignStatus.RUNNING else Stage.DONE.value
    )
    if single_process:
        driver_log = [
            f"[{now_iso()}] engine launcher: /usr/local/bin/tmd-submit",
            f"[{now_iso()}] run: single-process engine (GPU chosen by the launcher)",
        ]
    else:
        driver_log = [
            f"[{now_iso()}] [gpu {i % 2}] done  {n}"
            for i, n in enumerate(leg_names[:6])
        ]
    (run_dir / "stdout.log").write_text("\n".join(driver_log) + "\n")
    (run_dir / "stderr.log").write_text("")

    _write_tables(campaign, engine, edges)
    _write_db_row(campaign, runs_root, status)
    return campaign


def _write_tables(campaign: Campaign, engine: str, edges) -> None:
    """Result tables in whichever dialect the engine uses."""
    random.seed(11)
    rows = []
    for a, b in edges:
        rows.append((a, b, round(random.uniform(-2.5, 2.5), 3), round(random.uniform(0.1, 0.4), 3)))
    # One deliberate outlier so the cycle-closure view has something to find.
    if rows:
        a, b, _, u = rows[-1]
        rows[-1] = (a, b, 7.5, u)

    dg_rows = [
        (lig, round(random.uniform(-11, -6), 3), round(random.uniform(0.1, 0.5), 3))
        for lig in _LIGANDS
    ]

    if engine == "tmd":
        out = campaign.results_dir
        (out / "ddg_results.csv").write_text(
            "ligand_i,ligand_j,ddg,ddg_error\n"
            + "\n".join(f"{a},{b},{v},{u}" for a, b, v, u in rows) + "\n"
        )
        (out / "dg_results.csv").write_text(
            "ligand,dg,dg_error\n"
            + "\n".join(f"{l},{v},{u}" for l, v, u in dg_rows) + "\n"
        )
    else:
        out = campaign.gathered_dir
        (out / "ddg.tsv").write_text(
            "ligand_i\tligand_j\tDDG(i->j) (kcal/mol)\tuncertainty (kcal/mol)\n"
            + "\n".join(f"{a}\t{b}\t{v}\t{u}" for a, b, v, u in rows) + "\n"
        )
        (out / "dg.tsv").write_text(
            "ligand\tDG(MLE) (kcal/mol)\tuncertainty (kcal/mol)\n"
            + "\n".join(f"{l}\t{v}\t{u}" for l, v, u in dg_rows) + "\n"
        )


def _write_db_row(campaign: Campaign, runs_root: Path, status: CampaignStatus) -> None:
    db_path = runs_root / "state.db"
    _db.init_db(db_path)
    existing = _db.get_campaign(campaign.campaign_id, db_path)
    if existing is None:
        _db.insert_campaign(campaign, db_path)
    # A pid that is certainly dead: exercises the poller's "driver died"
    # classification without needing a real process to kill.
    _db.mark_running(campaign.campaign_id, 999_999, db_path)
    if status.is_terminal:
        _db.mark_terminal(campaign.campaign_id, status, db_path, exit_code=0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument(
        "--engine", action="append", default=None,
        help="engine name to fabricate (repeatable). Default: openfe and tmd.",
    )
    args = ap.parse_args()

    engines = args.engine or ["openfe", "tmd"]
    args.runs_root.mkdir(parents=True, exist_ok=True)

    for engine in engines:
        c = make_campaign(
            args.runs_root,
            engine=engine,
            name=f"fixture {engine} tyk2",
            status=CampaignStatus.RUNNING,
        )
        print(f"wrote {c.run_dir}")

    # One finished campaign, so the Results page has a completed subject.
    c = make_campaign(
        args.runs_root,
        engine="openfe",
        campaign_id="fixture_openfe_finished",
        name="fixture openfe finished",
        status=CampaignStatus.FINISHED,
    )
    print(f"wrote {c.run_dir}")
    print(f"\nPoint the dashboard at it:\n  FEPDASH_RUNS_ROOT={args.runs_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
