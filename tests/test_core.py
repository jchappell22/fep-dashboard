"""Tests for the parts that can be wrong without anyone noticing.

The emphasis is deliberate. This dashboard was built without access to the
box it runs on, so the tests concentrate on the logic that would fail
*silently* -- a dropped flag, a mis-mapped results column, a claim that lets
two GPUs onto one leg -- rather than on Streamlit rendering.

Run: pytest tests/ -q
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fepdash.core import gpu as _gpu  # noqa: E402
from fepdash.core import results as _results  # noqa: E402
from fepdash.core import state as _state  # noqa: E402
from fepdash.core.driver import render_driver  # noqa: E402
from fepdash.core.engines.base import (  # noqa: E402
    TemplateError,
    build_variables,
    load_engines,
    render,
)
from fepdash.core.models import Campaign, LegStatus, Method  # noqa: E402

ENGINES_DIR = Path(__file__).resolve().parents[1] / "engines"


@pytest.fixture
def engines():
    return load_engines(ENGINES_DIR)


def make_campaign(tmp_path: Path, engine="openfe", method=Method.RBFE, **params):
    run_dir = tmp_path / "campaign"
    for sub in ("plan/transformations", "legs", "work", "results", "gathered"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return Campaign(
        campaign_id="test",
        name="test",
        engine=engine,
        method=method,
        protein=tmp_path / "p.pdb",
        ligands=tmp_path / "l.sdf",
        gpus=[0, 1],
        params=params or {"n_repeats": 3, "n_cores": 4},
        run_dir=run_dir,
    )


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


def test_all_shipped_engines_load(engines):
    """A typo in a shipped TOML must not reach the box."""
    assert set(engines) == {"openfe", "tmd"}
    for engine in engines.values():
        assert engine.methods, f"{engine.name} failed to parse: {engine.label}"


def test_unknown_placeholder_raises_not_renders_empty():
    """The single most important guarantee in the whole template layer.

    A missing value must stop the launch. If it rendered as "" instead, a
    dropped `--pdb_path` would mean a twelve-hour run against no protein
    that still exits 0.
    """
    with pytest.raises(TemplateError) as exc:
        render("run --pdb {protein} --sdf {ligands}", {"protein": "p.pdb"}, where="t")
    assert "ligands" in str(exc.value)


def test_paths_with_spaces_are_quoted(tmp_path, engines):
    campaign = make_campaign(tmp_path)
    campaign.protein = Path("/data/my protein.pdb")
    script = render_driver(campaign, engines["openfe"])
    assert "'/data/my protein.pdb'" in script
    # And the per-leg bash variables stay quoted too.
    assert '"$tf"' in script and '"$name"' in script


def test_campaign_paths_are_relocatable(tmp_path, engines):
    """driver.sh must work after its campaign directory is moved."""
    script = render_driver(make_campaign(tmp_path), engines["openfe"])
    assert '"$CAMPAIGN_DIR/results"' in script
    assert str(tmp_path / "campaign" / "results") not in script


@pytest.mark.parametrize(
    "engine_name,method",
    [("openfe", Method.RBFE), ("openfe", Method.ABFE),
     ("tmd", Method.RBFE), ("tmd", Method.EDGE)],
)
def test_every_driver_variant_is_valid_bash(tmp_path, engines, engine_name, method):
    campaign = make_campaign(
        tmp_path, engine=engine_name, method=method,
        n_repeats=3, n_cores=4, ligand_a="a", ligand_b="b",
    )
    script = tmp_path / "driver.sh"
    script.write_text(render_driver(campaign, engines[engine_name]))
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0


def test_tmd_driver_never_passes_bg(tmp_path, engines):
    """--bg would make the tracked pid exit instantly and every campaign
    report 'finished' while the real work continued."""
    script = render_driver(
        make_campaign(tmp_path, engine="tmd", method=Method.RBFE), engines["tmd"]
    )
    assert "--bg" not in script


def test_tmd_driver_does_not_pin_gpus(tmp_path, engines):
    """tmd-submit picks the most-free card; overriding it breaks sharing."""
    script = render_driver(
        make_campaign(tmp_path, engine="tmd", method=Method.RBFE), engines["tmd"]
    )
    assert "export CUDA_VISIBLE_DEVICES" not in script


def test_openfe_driver_does_pin_gpus(tmp_path, engines):
    script = render_driver(make_campaign(tmp_path), engines["openfe"])
    assert 'export CUDA_VISIBLE_DEVICES="$gpu"' in script


def test_openfe_version_pin_is_exported(tmp_path, engines):
    campaign = make_campaign(tmp_path, n_repeats=3, n_cores=4, engine_version="1.8.1")
    assert 'export OPENFE_VERSION="1.8.1"' in render_driver(campaign, engines["openfe"])


def test_no_version_pin_means_no_export(tmp_path, engines):
    campaign = make_campaign(tmp_path, n_repeats=3, n_cores=4, engine_version="")
    assert "OPENFE_VERSION" not in render_driver(campaign, engines["openfe"])


def test_optional_flags_absent_when_unset(tmp_path, engines):
    variables = build_variables(make_campaign(tmp_path), engines["openfe"])
    assert variables["opt_cofactors"] == ""
    assert variables["opt_settings"] == ""


def test_optional_flags_present_when_set(tmp_path, engines):
    campaign = make_campaign(tmp_path)
    campaign.cofactors = tmp_path / "cof.sdf"
    variables = build_variables(campaign, engines["openfe"])
    assert variables["opt_cofactors"].startswith(" -C ")


# ---------------------------------------------------------------------------
# Leg state
# ---------------------------------------------------------------------------


def test_unstarted_legs_show_as_pending(tmp_path, engines):
    """A campaign in its first minutes must not look empty."""
    campaign = make_campaign(tmp_path)
    for name in ("rbfe_a_complex_b_complex", "rbfe_a_solvent_b_solvent"):
        (campaign.plan_dir / "transformations" / f"{name}.json").write_text("{}")
    prog = _state.progress(campaign, "transformations/*.json")
    assert prog.total == 2
    assert prog.pending == 2


def test_torn_status_json_does_not_crash(tmp_path, engines):
    campaign = make_campaign(tmp_path)
    leg = campaign.legs_dir / "broken"
    leg.mkdir(parents=True)
    (leg / "status.json").write_text('{"leg_id": "broken", "stat')  # truncated
    assert _state.scan_legs(campaign) == []


def test_eta_is_none_before_any_leg_completes(tmp_path):
    campaign = make_campaign(tmp_path)
    (campaign.legs_dir / "x").mkdir(parents=True)
    (campaign.legs_dir / "x" / "status.json").write_text(
        json.dumps({"leg_id": "x", "status": "running"})
    )
    prog = _state.progress(campaign, "")
    assert prog.eta(2) is None, "an ETA from no data is worse than no ETA"


def test_leg_edge_and_type_parsing():
    from fepdash.core.models import Leg

    leg = Leg(leg_id="rbfe_ligA_complex_ligB_complex", status=LegStatus.DONE)
    assert leg.leg_type == "complex"
    leg2 = Leg(leg_id="something_unrecognised", status=LegStatus.DONE)
    assert leg2.edge == "something_unrecognised"  # unchanged, not mangled


def test_tail_reads_only_the_end_of_a_large_log(tmp_path):
    big = tmp_path / "big.log"
    big.write_text("\n".join(f"line {i}" for i in range(200_000)))
    out = _state.tail(big, n=5)
    assert out.count("\n") == 4
    assert "line 199999" in out


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def test_openfe_tsv_and_tmd_csv_normalise_identically(tmp_path):
    tsv = tmp_path / "ddg.tsv"
    tsv.write_text(
        "ligand_i\tligand_j\tDDG(i->j) (kcal/mol)\tuncertainty (kcal/mol)\n"
        "a\tb\t-1.5\t0.2\n"
    )
    csv = tmp_path / "ddg_results.csv"
    csv.write_text("ligand_i,ligand_j,ddg,ddg_error\na,b,-1.5,0.2\n")

    from_tsv = _results.load_ddg(tsv, "tsv")
    from_csv = _results.load_ddg(csv, "csv")
    assert from_tsv.ok and from_csv.ok
    assert list(from_tsv.normalised.columns) == list(from_csv.normalised.columns)
    assert from_tsv.normalised.iloc[0]["ddG"] == from_csv.normalised.iloc[0]["ddG"]


def test_unrecognised_columns_report_rather_than_guess(tmp_path):
    """A mis-mapped dG column yields a plausible-looking, wrong ranking.
    Better to show the raw table and say so."""
    path = tmp_path / "weird.tsv"
    path.write_text("alpha\tbeta\tgamma\n1\t2\t3\n")
    table = _results.load_dg(path, "tsv")
    assert not table.ok
    assert table.problems and "could not find" in table.problems[0]


def test_dg_ranking_is_ascending(tmp_path):
    path = tmp_path / "dg.tsv"
    path.write_text("ligand\tDG(MLE) (kcal/mol)\tuncertainty\nw\t-5\t0.1\nx\t-9\t0.1\n")
    table = _results.load_dg(path, "tsv")
    assert table.normalised.iloc[0]["ligand"] == "x", "tightest binder ranks first"


def test_cycle_closure_detects_an_inconsistent_triangle():
    import pandas as pd

    # A triangle that does not close: a->b 1, b->c 1, a->c 5.
    ddg = pd.DataFrame(
        {
            "ligand_i": ["a", "b", "a"],
            "ligand_j": ["b", "c", "c"],
            "ddG": [1.0, 1.0, 5.0],
            "uncertainty": [0.1, 0.1, 0.1],
        }
    )
    out = _results.cycle_closure(ddg)
    assert out is not None
    assert out["residual"].abs().max() > 0.5, "a 3 kcal/mol misclosure must show up"


def test_cycle_closure_spreads_blame_around_a_bare_triangle():
    """Documents a real limitation, so nobody reads the top row as 'the' culprit.

    Least squares cannot tell which member of an inconsistent cycle is
    wrong, so in a bare triangle it distributes the discrepancy equally.
    Blame only concentrates when an edge sits in several overlapping cycles.
    """
    import pandas as pd

    ddg = pd.DataFrame(
        {
            "ligand_i": ["a", "b", "a"],
            "ligand_j": ["b", "c", "c"],
            "ddG": [1.0, 1.0, 5.0],
            "uncertainty": [0.1, 0.1, 0.1],
        }
    )
    residuals = _results.cycle_closure(ddg)["residual"].abs()
    assert residuals.max() - residuals.min() < 1e-9


def test_cycle_closure_returns_none_for_a_tree():
    import pandas as pd

    ddg = pd.DataFrame(
        {"ligand_i": ["a", "b"], "ligand_j": ["b", "c"],
         "ddG": [1.0, 1.0], "uncertainty": [0.1, 0.1]}
    )
    assert _results.cycle_closure(ddg) is None


# ---------------------------------------------------------------------------
# GPU claims
# ---------------------------------------------------------------------------


def test_claim_is_all_or_nothing(tmp_path):
    _gpu.claim_gpus(tmp_path, [0, 1], "campaign-a", os.getpid())
    with pytest.raises(_gpu.GpuBusy):
        _gpu.claim_gpus(tmp_path, [1, 2], "campaign-b", os.getpid())
    # campaign-b must not have partially claimed GPU 2.
    assert 2 not in _gpu.read_claims(tmp_path)


def test_dead_owner_claim_is_swept(tmp_path):
    _gpu.claim_gpus(tmp_path, [0], "dead-campaign", 999_999)
    assert _gpu.read_claims(tmp_path) == {}


def test_unwritable_runs_root_is_reported_not_raised(tmp_path):
    """An unwritable runs_root must produce an actionable message naming the
    config, not a PermissionError out of pathlib. This is the most likely
    first-run failure on a shared box."""
    from fepdash.core.config import Config, check_runs_root

    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)  # r-x: cannot create anything inside
    try:
        cfg = Config(
            runs_root=locked / "runs",
            db_path=locked / "runs" / "state.db",
            engines_dir=ENGINES_DIR,
            inputs_root=tmp_path,
            scripts_dir=tmp_path,
            allowed_gpus=(0,),
        )
        problem = check_runs_root(cfg)
        assert problem is not None
        assert "runs_root" in problem
    finally:
        locked.chmod(0o700)


def test_writable_runs_root_passes(tmp_path):
    from fepdash.core.config import Config, check_runs_root

    cfg = Config(
        runs_root=tmp_path / "runs",  # does not exist yet, but creatable
        db_path=tmp_path / "runs" / "state.db",
        engines_dir=ENGINES_DIR,
        inputs_root=tmp_path,
        scripts_dir=tmp_path,
        allowed_gpus=(0,),
    )
    assert check_runs_root(cfg) is None


def test_moved_campaign_is_found_under_runs_root(tmp_path):
    """A relocated runs tree must not orphan every campaign.

    run_dir is recorded absolute. Restore a backup to a different mount, or
    move runs_root, and every row points somewhere that no longer exists --
    campaigns would render with no legs and no results while their
    directories sat intact under the new root.
    """
    from fepdash.core import db as _db

    old_root = tmp_path / "old"
    new_root = tmp_path / "new"
    (new_root / "c1").mkdir(parents=True)

    row = {
        "campaign_id": "c1",
        "name": "c1",
        "engine": "openfe",
        "method": "rbfe",
        "run_dir": str(old_root / "c1"),  # no longer exists
        "gpus": [0],
        "created_at": "",
        "manifest_json": {},
    }
    campaign = _db.campaign_from_row(row, runs_root=new_root)
    assert campaign.run_dir == new_root / "c1"

    # Without runs_root, the recorded path is kept -- no silent guessing.
    assert _db.campaign_from_row(row).run_dir == old_root / "c1"


def test_zombie_process_is_not_alive(tmp_path):
    """The bug that leaves cards claimed forever.

    The dashboard never wait()s on a driver, so a driver that exits while
    Streamlit is up becomes a zombie. It still answers kill(pid, 0). If that
    counted as alive, the campaign would sit at 'running' forever and its
    GPUs would never be released. Must hold on Linux AND macOS -- this was
    originally /proc-only, so it passed in production and failed in dev.
    """
    proc = subprocess.Popen(["true"])
    proc.poll()  # do NOT wait(): leave it defunct
    for _ in range(100):
        if not _gpu._pid_alive(proc.pid):
            break
        import time

        time.sleep(0.05)
    else:
        proc.wait()
        pytest.fail("a zombie was reported alive; GPU claims would leak")
    proc.wait()


def test_release_frees_only_its_own(tmp_path):
    _gpu.claim_gpus(tmp_path, [0], "a", os.getpid())
    _gpu.claim_gpus(tmp_path, [1], "b", os.getpid())
    _gpu.release_gpus(tmp_path, "a")
    claims = _gpu.read_claims(tmp_path)
    assert 0 not in claims and 1 in claims
