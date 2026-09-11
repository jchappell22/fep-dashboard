"""Tests for the input checks.

These matter because the checker's job is to be trusted: a false clean bill
of health is worse than no checker at all, since someone then commits a day
of GPU time on its say-so. So each test builds a file with exactly one
defect and asserts that defect -- and only work that would really fail is
an ERROR.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fepdash.core import prep_check  # noqa: E402
from fepdash.core.prep_check import Severity  # noqa: E402


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def pdb_atom(
    serial=1, name=" CA ", alt=" ", res="ALA", chain="A", resnum=1,
    x=0.0, y=0.0, z=0.0, element=" C", record="ATOM  ",
):
    """One PDB record, column-exact -- the parser is column-based."""
    return (
        f"{record}{serial:>5d} {name:<4s}{alt}{res:>3s} {chain}{resnum:>4d}    "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          {element:>2s}"
    )


def write_pdb(tmp_path, lines, filename="p.pdb"):
    path = tmp_path / filename
    path.write_text("\n".join(lines) + "\nEND\n")
    return path


def protein_lines(n_res=4, with_h=True, chain="A", res="ALA", start=1):
    lines, serial = [], 1
    for i in range(n_res):
        for nm, el in ((" N  ", " N"), (" CA ", " C"), (" C  ", " C"), (" O  ", " O")):
            lines.append(pdb_atom(serial, nm, res=res, chain=chain,
                                  resnum=start + i, element=el))
            serial += 1
        if with_h:
            # Enough hydrogens to clear the "partially protonated" heuristic.
            for j in range(5):
                lines.append(pdb_atom(serial, f" H{j}", res=res, chain=chain,
                                      resnum=start + i, element=" H"))
                serial += 1
    return lines


def sdf_record(name="lig1", n_atoms=3, hydrogens=2, z=1.0, charge_block=""):
    atoms = []
    for i in range(n_atoms - hydrogens):
        atoms.append(f"    0.0000    0.0000{z:>10.4f} C   0  0  0  0  0  0  0  0  0  0  0  0")
    for i in range(hydrogens):
        atoms.append(f"    0.0000    0.0000{z:>10.4f} H   0  0  0  0  0  0  0  0  0  0  0  0")
    return "\n".join(
        [name, "  fepdash", "", f"{n_atoms:>3d}  0  0  0  0  0  0  0  0  0999 V2000"]
        + atoms
        + ([charge_block] if charge_block else [])
        + ["M  END", "$$$$"]
    )


def write_sdf(tmp_path, records, filename="l.sdf"):
    path = tmp_path / filename
    path.write_text("\n".join(records) + "\n")
    return path


def titles(report):
    return " | ".join(f.title for f in report.findings)


# ---------------------------------------------------------------------------
# Protein
# ---------------------------------------------------------------------------


def test_clean_protein_passes(tmp_path):
    report = prep_check.check_protein(write_pdb(tmp_path, protein_lines()))
    assert report.ok, titles(report)
    assert not report.errors


def test_not_a_pdb_is_an_error(tmp_path):
    path = tmp_path / "x.pdb"
    path.write_text("this is not a pdb\n")
    report = prep_check.check_protein(path)
    assert not report.ok
    assert "No ATOM or HETATM" in titles(report)


def test_missing_hydrogens_warns_but_does_not_block(tmp_path):
    """OpenMM will add them -- the risk is that it picks the states, which is
    a scientific decision, not a broken file."""
    report = prep_check.check_protein(
        write_pdb(tmp_path, protein_lines(with_h=False))
    )
    assert report.ok, "missing hydrogens must not be a hard error"
    assert any(f.severity is Severity.WARN for f in report.findings)
    assert "hydrogens" in titles(report).lower()


def test_altloc_is_an_error(tmp_path):
    lines = protein_lines()
    lines.append(pdb_atom(999, " CB ", alt="A", res="ALA", resnum=1, element=" C"))
    lines.append(pdb_atom(1000, " CB ", alt="B", res="ALA", resnum=1, element=" C"))
    report = prep_check.check_protein(write_pdb(tmp_path, lines))
    assert not report.ok
    assert "Alternate conformations" in titles(report)


def test_ligand_left_in_the_pdb_is_an_error(tmp_path):
    """A ligand left in the protein PDB -- the most common fatal mistake."""
    lines = protein_lines()
    lines.append(
        pdb_atom(999, " C1 ", res="LIG", resnum=900, element=" C", record="HETATM")
    )
    report = prep_check.check_protein(write_pdb(tmp_path, lines))
    assert not report.ok
    assert "non-standard HETATM" in titles(report)


def test_terminal_caps_are_accepted(tmp_path):
    """Regression: NMA is a correctly capped C-terminus, spelled the way
    tools other than Amber write NME. A real prepped protein from a
    successful campaign carried it, and flagging it as an error would teach
    people to ignore the checker."""
    for cap in ("NME", "NMA", "ACE", "NHE"):
        lines = protein_lines() + protein_lines(n_res=1, res=cap, start=99)
        report = prep_check.check_protein(write_pdb(tmp_path, lines, f"{cap}.pdb"))
        assert report.ok, f"{cap} should not be an error: {titles(report)}"
        assert "unrecognised" not in titles(report), cap


def test_unknown_atom_residue_warns_rather_than_errors(tmp_path):
    """The residue list cannot be exhaustive, so an unrecognised ATOM
    residue is 'check this', not 'this will fail'."""
    lines = protein_lines() + protein_lines(n_res=1, res="SEP", start=77)
    report = prep_check.check_protein(write_pdb(tmp_path, lines))
    assert report.ok, "must not hard-block on a possibly-fine modified residue"
    assert "unrecognised residue name" in titles(report)


def test_waters_are_info_not_error(tmp_path):
    lines = protein_lines()
    lines.append(
        pdb_atom(999, " O  ", res="HOH", resnum=500, element=" O", record="HETATM")
    )
    report = prep_check.check_protein(write_pdb(tmp_path, lines))
    assert report.ok, "waters are a choice, not a failure"
    assert any(f.severity is Severity.INFO for f in report.findings)


def test_common_ions_are_not_flagged_as_unknown(tmp_path):
    lines = protein_lines()
    lines.append(
        pdb_atom(999, "ZN  ", res="ZN", resnum=600, element="ZN", record="HETATM")
    )
    report = prep_check.check_protein(write_pdb(tmp_path, lines))
    assert "forcefield does not know" not in titles(report)


def test_chain_break_warns(tmp_path):
    lines = protein_lines(n_res=3, start=1) + protein_lines(n_res=3, start=50)
    report = prep_check.check_protein(write_pdb(tmp_path, lines))
    assert "chain break" in titles(report).lower()
    assert report.ok, "a gap is a warning, not a blocker"


def test_multiple_chains_reported(tmp_path):
    lines = protein_lines(chain="A") + protein_lines(chain="B")
    report = prep_check.check_protein(write_pdb(tmp_path, lines))
    assert report.facts["chains"] == ["A", "B"]


def test_element_falls_back_to_atom_name():
    """Columns 77-78 are often blank in hand-made or legacy files."""
    line = pdb_atom(1, " CA ", element="  ")
    assert prep_check._element(line) == "C"
    assert prep_check._element(pdb_atom(1, "1HB ", element="  ")) == "H"


# ---------------------------------------------------------------------------
# Ligands
# ---------------------------------------------------------------------------


def test_clean_sdf_passes(tmp_path):
    path = write_sdf(tmp_path, [sdf_record(f"lig{i}") for i in range(4)])
    report = prep_check.check_ligands(path)
    assert report.ok, titles(report)
    assert report.facts["molecules"] == 4


def test_empty_sdf_is_an_error(tmp_path):
    path = tmp_path / "l.sdf"
    path.write_text("")
    report = prep_check.check_ligands(path)
    assert not report.ok


def test_2d_coordinates_are_an_error(tmp_path):
    """RBFE does not dock -- a 2D depiction cannot be perturbed in place."""
    path = write_sdf(
        tmp_path, [sdf_record(f"lig{i}", z=0.0) for i in range(3)]
    )
    report = prep_check.check_ligands(path)
    assert not report.ok
    assert "3D coordinates" in titles(report)


def test_missing_hydrogens_is_an_error(tmp_path):
    path = write_sdf(
        tmp_path, [sdf_record(f"lig{i}", n_atoms=3, hydrogens=0) for i in range(3)]
    )
    report = prep_check.check_ligands(path)
    assert not report.ok
    assert "explicit hydrogens" in titles(report)


def test_duplicate_names_are_an_error(tmp_path):
    path = write_sdf(tmp_path, [sdf_record("same"), sdf_record("same"), sdf_record("x")])
    report = prep_check.check_ligands(path)
    assert not report.ok
    assert "duplicated" in titles(report).lower()


def test_blank_name_is_an_error(tmp_path):
    path = write_sdf(tmp_path, [sdf_record(""), sdf_record("b"), sdf_record("c")])
    report = prep_check.check_ligands(path)
    assert not report.ok
    assert "no name" in titles(report)


def test_mixed_net_charge_warns(tmp_path):
    """The quiet one: charge-changing edges run and give incomparable numbers,
    because explicit_charge_correction is off by default."""
    charged = sdf_record("anion", charge_block="M  CHG  1   1  -1")
    path = write_sdf(tmp_path, [sdf_record("neutral"), charged, sdf_record("c")])
    report = prep_check.check_ligands(path)
    assert "Mixed net charges" in titles(report)
    assert report.facts["charges"] == [-1, 0]


def test_m_chg_overrides_legacy_charge_column(tmp_path):
    """M CHG supersedes the legacy column; counting both would double up."""
    lines = sdf_record("x", charge_block="M  CHG  1   1  -1").splitlines()
    n_atoms = 3
    assert prep_check._v2000_charge(lines, n_atoms) == -1


def test_size_spread_warns(tmp_path):
    path = write_sdf(
        tmp_path,
        [sdf_record("small", n_atoms=5, hydrogens=2),
         sdf_record("huge", n_atoms=60, hydrogens=20),
         sdf_record("mid", n_atoms=10, hydrogens=4)],
    )
    report = prep_check.check_ligands(path)
    assert "spread in molecule size" in titles(report)


def test_v3000_is_parsed(tmp_path):
    block = "\n".join([
        "ligv3", "  fepdash", "",
        "  0  0  0  0  0  0            999 V3000",
        "M  V30 BEGIN CTAB",
        "M  V30 COUNTS 3 2 0 0 0",
        "M  V30 BEGIN ATOM",
        "M  V30 1 C 0.0 0.0 1.5 0",
        "M  V30 2 H 0.0 0.0 2.5 0",
        "M  V30 3 O 0.0 0.0 0.5 0 CHG=-1",
        "M  V30 END ATOM",
        "M  V30 END CTAB",
        "M  END", "$$$$",
    ])
    report = prep_check.check_ligands(write_sdf(tmp_path, [block]))
    assert report.facts["molecules"] == 1
    assert report.facts["charges"] == [-1]
    assert "3D coordinates" not in titles(report)


def test_unreadable_file_reports_rather_than_raises(tmp_path):
    missing = tmp_path / "nope.sdf"
    report = prep_check.check_ligands(missing)
    assert not report.ok
    assert "Cannot read" in titles(report)
