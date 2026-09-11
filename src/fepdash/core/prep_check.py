"""fepdash.core.prep_check -- cheap sanity checks on protein and ligand files.

Why this is plain text parsing and not RDKit
--------------------------------------------

The dashboard's env is streamlit + pandas and nothing else, deliberately --
it has to stand up on a shared box without fighting package installs. RDKit
and OpenMM live in the engine envs, which the dashboard only ever reaches
through their launchers.

So these checks read the files as text. That is a real limitation and the UI
says so: this catches the structural mistakes that kill a campaign in its
first minutes (no hydrogens, 2D coordinates, duplicate molecule names,
mixed net charge), not chemistry mistakes (wrong tautomer, bad protonation
state, mis-perceived bond orders). Those need a real toolkit and a human.

Every check here is one the author has seen waste GPU hours. The ordering of
:class:`Severity` reflects that: ERROR means the run will fail or be
meaningless; WARN means look before you commit a day of cards.
"""

from __future__ import annotations

import enum
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


class Severity(str, enum.Enum):
    ERROR = "error"  # this will fail, or produce a meaningless answer
    WARN = "warn"  # probably wrong; look before spending cards
    INFO = "info"  # worth knowing


@dataclass
class Finding:
    severity: Severity
    title: str
    detail: str = ""
    fix: str = ""


@dataclass
class Report:
    path: Path
    kind: str  # "protein" | "ligands"
    findings: list[Finding] = field(default_factory=list)
    facts: dict = field(default_factory=dict)

    def add(self, severity: Severity, title: str, detail: str = "", fix: str = "") -> None:
        self.findings.append(Finding(severity, title, detail, fix))

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.WARN]

    @property
    def ok(self) -> bool:
        return not self.errors


# ---------------------------------------------------------------------------
# Protein
# ---------------------------------------------------------------------------

# Residues the default forcefield set (ff14SB + tip3p + phosaa10 + lipid17)
# can parameterise without extra work. Anything else needs its own
# parameters or has to come out of the PDB.
_STANDARD_RESIDUES = {
    # amino acids, including the protonation-state variants ff14SB knows
    "ALA", "ARG", "ASN", "ASP", "ASH", "CYS", "CYX", "CYM", "GLN", "GLU",
    "GLH", "GLY", "HIS", "HID", "HIE", "HIP", "ILE", "LEU", "LYS", "LYN",
    "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "HYP",
    # Terminal caps. Spellings vary by tool -- Amber writes NME, other
    # packages write NMA for the same N-methylamide cap -- so accept both
    # rather than flagging a correctly capped protein.
    "ACE", "NME", "NMA", "NHE", "NH2",
    # water and common ions
    "HOH", "WAT", "NA", "CL", "K", "MG", "ZN", "CA", "SOD", "CLA",
}

_IONS = {"NA", "CL", "K", "MG", "ZN", "CA", "SOD", "CLA", "MN", "FE", "CU"}
_WATERS = {"HOH", "WAT", "TIP", "TIP3", "SOL"}


def check_protein(path: Path) -> Report:
    """Structural checks on a PDB destined for OpenFE or TMD."""
    report = Report(path=path, kind="protein")

    try:
        text = path.read_text(errors="replace")
    except OSError as exc:
        report.add(Severity.ERROR, "Cannot read the file", str(exc))
        return report

    lines = text.splitlines()
    atoms = [l for l in lines if l.startswith(("ATOM  ", "HETATM"))]
    if not atoms:
        report.add(
            Severity.ERROR,
            "No ATOM or HETATM records",
            "This does not look like a PDB file.",
            "Export a PDB (not mmCIF, not a Maestro .mae) from your prep tool.",
        )
        return report

    hetatms = [l for l in atoms if l.startswith("HETATM")]
    elements = Counter(_element(l) for l in atoms)
    resnames = Counter(l[17:20].strip().upper() for l in atoms)
    chains = sorted({l[21] for l in atoms if l[21].strip()})
    altlocs = {l[16] for l in atoms if l[16].strip()}

    report.facts = {
        "atoms": len(atoms),
        "hetatms": len(hetatms),
        "chains": chains,
        "residues": sum(
            1 for _ in {(l[21], l[22:27]) for l in atoms if not l.startswith("HETATM")}
        ),
    }

    # -- hydrogens --------------------------------------------------------
    n_h = elements.get("H", 0)
    report.facts["hydrogens"] = n_h
    if n_h == 0:
        report.add(
            Severity.WARN,
            "No hydrogens in the structure",
            "OpenMM can add them at setup, but it will use its own default "
            "protonation states -- which means His tautomers and the "
            "charge on Asp/Glu/Lys are chosen for you, not by you.",
            "Protonate deliberately (PrepWizard, PDB2PQR, or pdb4amber) at "
            "the pH you care about, especially for any residue in the pocket.",
        )
    elif n_h < len(atoms) * 0.3:
        report.add(
            Severity.WARN,
            f"Only {n_h} hydrogens for {len(atoms)} atoms",
            "That is well below the ~50% you would expect for a fully "
            "protonated structure, so the file may be partially protonated.",
            "Re-run protonation over the whole structure.",
        )

    # -- alternate locations ----------------------------------------------
    if altlocs:
        report.add(
            Severity.ERROR,
            f"Alternate conformations present (altLoc {', '.join(sorted(altlocs))})",
            "Two sets of coordinates for the same atoms. Most setup paths "
            "either fail or silently keep whichever comes first.",
            "Keep one altLoc (usually A) and strip the rest before running.",
        )

    # -- waters and ions --------------------------------------------------
    n_water = sum(c for r, c in resnames.items() if r in _WATERS)
    if n_water:
        report.add(
            Severity.INFO,
            f"{n_water} water atoms present",
            "Crystallographic waters are usually stripped -- the protocol "
            "solvates the system itself with a 1.5 nm TIP3P padding. Keep "
            "them only if a specific water is known to bridge the ligand.",
            "Strip with your prep tool if you have no reason to keep them.",
        )

    # -- non-standard residues -------------------------------------------
    #
    # Split by record type rather than lumping them together. An unknown
    # HETATM is almost always a real blocker -- a ligand, cofactor, or
    # buffer molecule with no parameters. An unknown ATOM residue is far
    # more often a modified or capped residue whose spelling is simply not
    # in the list below (Amber writes NME for the cap other tools call NMA),
    # and this list cannot be exhaustive.
    #
    # Calling the second case an ERROR would flag correctly prepared
    # proteins, and a checker that cries wolf gets ignored -- which costs
    # more than the check ever saved.
    unknown_het: dict[str, int] = {}
    unknown_atom: dict[str, int] = {}
    for line in atoms:
        name = line[17:20].strip().upper()
        if name in _STANDARD_RESIDUES or name in _WATERS or name in _IONS:
            continue
        bucket = unknown_het if line.startswith("HETATM") else unknown_atom
        bucket[name] = bucket.get(name, 0) + 1

    if unknown_het:
        listed = ", ".join(f"{r} ({c} atoms)" for r, c in sorted(unknown_het.items())[:8])
        report.add(
            Severity.ERROR,
            f"{len(unknown_het)} non-standard HETATM residue(s)",
            f"{listed}. The default forcefield set is ff14SB + TIP3P + "
            f"phosaa10 + lipid17, which covers standard amino acids, water "
            f"and common ions. A cofactor, buffer molecule, or a ligand left "
            f"in the file has no parameters, and setup fails.",
            "Remove it, or pass it separately as a cofactor SDF so it gets "
            "small-molecule parameters. The ligand you are perturbing must "
            "NOT be in the protein PDB.",
        )

    if unknown_atom:
        listed = ", ".join(f"{r} ({c} atoms)" for r, c in sorted(unknown_atom.items())[:8])
        report.add(
            Severity.WARN,
            f"{len(unknown_atom)} unrecognised residue name(s) in ATOM records",
            f"{listed}. These are probably modified or capped residues. The "
            f"list this check uses is not exhaustive, so this may well be "
            f"fine -- a correctly capped C-terminus is written NME by Amber "
            f"and NMA by other tools, for instance.",
            "Confirm ff14SB has a template for each one. If not, rename to "
            "the Amber spelling or supply parameters.",
        )

    # -- chain breaks ------------------------------------------------------
    gaps = _residue_gaps(atoms)
    if gaps:
        shown = ", ".join(f"{c}:{a}->{b}" for c, a, b in gaps[:6])
        report.add(
            Severity.WARN,
            f"{len(gaps)} apparent chain break(s)",
            f"{shown}. Residue numbering jumps, which usually means "
            f"unresolved loops. OpenMM will cap nothing automatically -- the "
            f"gap becomes two chain ends with charged termini in the middle "
            f"of your protein.",
            "Model the loops, or cap the termini, if the gap is anywhere near "
            "the binding site. A gap far from the pocket is usually tolerable.",
        )

    if len(chains) > 1:
        report.add(
            Severity.INFO,
            f"{len(chains)} chains: {', '.join(chains)}",
            "Every chain is simulated. Extra copies multiply cost for no "
            "benefit unless the site is at an interface.",
            "Keep only the chain(s) forming the binding site.",
        )

    return report


def _element(line: str) -> str:
    """Element symbol from a PDB ATOM/HETATM record.

    Prefers columns 77-78, the actual element field. Falls back to the atom
    name, where PDB convention puts the element in columns 13-14 and a
    leading digit means the name was left-justified.
    """
    symbol = line[76:78].strip().upper()
    if symbol:
        return symbol
    name = line[12:16].strip()
    if not name:
        return ""
    return (name[1] if name[0].isdigit() else name[0]).upper()


def _residue_gaps(atoms: list[str]) -> list[tuple[str, int, int]]:
    """Jumps in residue numbering within a chain, as (chain, from, to)."""
    seen: dict[str, list[int]] = {}
    for line in atoms:
        if line.startswith("HETATM"):
            continue
        chain = line[21]
        try:
            number = int(line[22:26])
        except ValueError:
            continue
        bucket = seen.setdefault(chain, [])
        if not bucket or bucket[-1] != number:
            bucket.append(number)

    gaps = []
    for chain, numbers in seen.items():
        ordered = sorted(set(numbers))
        for a, b in zip(ordered, ordered[1:]):
            if b - a > 1:
                gaps.append((chain, a, b))
    return gaps


# ---------------------------------------------------------------------------
# Ligands
# ---------------------------------------------------------------------------


@dataclass
class LigandRecord:
    name: str
    n_atoms: int
    n_hydrogens: int
    has_3d: bool
    charge: int
    index: int


def check_ligands(path: Path) -> Report:
    """Structural checks on an SDF destined for OpenFE or TMD."""
    report = Report(path=path, kind="ligands")

    try:
        text = path.read_text(errors="replace")
    except OSError as exc:
        report.add(Severity.ERROR, "Cannot read the file", str(exc))
        return report

    records = _parse_sdf(text)
    if not records:
        report.add(
            Severity.ERROR,
            "No molecules found",
            "Nothing parsed as an SDF record.",
            "Export a V2000/V3000 SDF. A .smi or .mol2 will not work here.",
        )
        return report

    report.facts = {
        "molecules": len(records),
        "charges": sorted({r.charge for r in records}),
    }

    # -- names -------------------------------------------------------------
    unnamed = [r.index for r in records if not r.name]
    if unnamed:
        report.add(
            Severity.ERROR,
            f"{len(unnamed)} molecule(s) have no name",
            f"First at record {unnamed[0] + 1}. Both engines key edges and "
            f"results by molecule title, and a blank one makes the results "
            f"table unreadable at best.",
            "Set the title line (the first line of each SDF record).",
        )

    duplicates = [n for n, c in Counter(r.name for r in records if r.name).items() if c > 1]
    if duplicates:
        report.add(
            Severity.ERROR,
            f"{len(duplicates)} duplicated molecule name(s)",
            f"{', '.join(duplicates[:6])}. Edges are identified by name, so "
            f"duplicates collide -- results get silently overwritten or "
            f"mismatched.",
            "Make every title unique.",
        )

    # -- geometry ----------------------------------------------------------
    flat = [r.name or f"#{r.index + 1}" for r in records if not r.has_3d]
    if flat:
        report.add(
            Severity.ERROR,
            f"{len(flat)} molecule(s) have no 3D coordinates",
            f"{', '.join(flat[:6])}. All z coordinates are zero, so these "
            f"are 2D depictions. RBFE needs every ligand posed in the "
            f"binding site.",
            "Generate 3D conformers and place them in the pocket -- align to "
            "a crystallographic pose or dock them, then minimise.",
        )

    no_h = [r.name or f"#{r.index + 1}" for r in records if r.n_hydrogens == 0]
    if no_h:
        report.add(
            Severity.ERROR,
            f"{len(no_h)} molecule(s) have no explicit hydrogens",
            f"{', '.join(no_h[:6])}. The OpenFF small-molecule forcefield "
            f"needs explicit hydrogens; implicit ones are not inferred here.",
            "Add hydrogens at the intended protonation state before export.",
        )

    # -- net charge --------------------------------------------------------
    charges = Counter(r.charge for r in records)
    if len(charges) > 1:
        summary = ", ".join(f"{c:+d} ({n} ligands)" for c, n in sorted(charges.items()))
        report.add(
            Severity.WARN,
            "Mixed net charges across the series",
            f"{summary}. A transformation that changes net charge needs "
            f"special handling -- openfe's `explicit_charge_correction` is "
            f"OFF by default, so charge-changing edges will run and give you "
            f"numbers that are not comparable.",
            "Either split into same-charge sub-series, or enable explicit "
            "charge correction (co-alchemical ion) before trusting any edge "
            "that crosses a charge boundary.",
        )

    # -- size spread -------------------------------------------------------
    sizes = [r.n_atoms for r in records]
    if sizes and max(sizes) - min(sizes) > 25:
        report.add(
            Severity.WARN,
            f"Large spread in molecule size ({min(sizes)}-{max(sizes)} atoms)",
            "RBFE assumes a congeneric series. Very different ligands map "
            "poorly, and a mapping with many dummy atoms converges slowly "
            "or not at all. TMD's docs put a hard ceiling of 30 dummy atoms "
            "per transformation.",
            "Check the network on the Runs page after planning, and consider "
            "splitting the set.",
        )

    if len(records) < 3:
        report.add(
            Severity.WARN,
            f"Only {len(records)} molecule(s)",
            "A relative network needs at least a few ligands to be worth "
            "planning; with two you get a single edge and no cycles.",
        )

    return report


def _parse_sdf(text: str) -> list[LigandRecord]:
    """Parse enough of an SDF to sanity-check it. V2000 and V3000.

    Deliberately tolerant: a record we cannot fully parse still contributes
    its name, because a partial answer beats refusing to look.
    """
    records: list[LigandRecord] = []
    blocks = re.split(r"^\$\$\$\$\s*$", text, flags=re.MULTILINE)

    for index, block in enumerate(blocks):
        # Strip EXACTLY the newline that terminated the preceding `$$$$`
        # line -- and only for blocks that follow one. The first block has
        # no such artifact, so a leading newline there is a genuine empty
        # title line.
        #
        # Getting this wrong eats the title line of an unnamed molecule,
        # shifting the record up by one and making it unparseable. The
        # molecule then vanishes from the report instead of being flagged,
        # which is the worst failure a checker can have: a clean bill of
        # health on a file that will not run.
        if index > 0:
            if block.startswith("\r\n"):
                block = block[2:]
            elif block.startswith("\n"):
                block = block[1:]

        lines = block.splitlines()
        if len(lines) < 4 or not any(l.strip() for l in lines):
            continue

        name = lines[0].strip()
        counts = lines[3] if len(lines) > 3 else ""

        if "V3000" in counts or any("V30 COUNTS" in l for l in lines[:8]):
            record = _parse_v3000(lines, name, index)
        else:
            record = _parse_v2000(lines, name, index, counts)
        if record is not None:
            records.append(record)

    return records


def _parse_v2000(
    lines: list[str], name: str, index: int, counts: str
) -> Optional[LigandRecord]:
    try:
        n_atoms = int(counts[0:3])
    except (ValueError, IndexError):
        return None
    if n_atoms <= 0:
        return None

    atom_lines = lines[4 : 4 + n_atoms]
    hydrogens = 0
    all_z_zero = True
    for line in atom_lines:
        try:
            z = float(line[20:30])
            symbol = line[31:34].strip()
        except (ValueError, IndexError):
            continue
        if symbol == "H":
            hydrogens += 1
        if abs(z) > 1e-6:
            all_z_zero = False

    return LigandRecord(
        name=name,
        n_atoms=n_atoms,
        n_hydrogens=hydrogens,
        has_3d=not all_z_zero,
        charge=_v2000_charge(lines, n_atoms),
        index=index,
    )


def _v2000_charge(lines: list[str], n_atoms: int) -> int:
    """Net formal charge.

    M  CHG lines win when present -- they are the modern encoding and they
    supersede the legacy per-atom charge column entirely.
    """
    total = 0
    found_chg = False
    for line in lines:
        if line.startswith("M  CHG"):
            found_chg = True
            fields = line.split()[3:]  # after "M", "CHG", count
            for value in fields[1::2]:
                try:
                    total += int(value)
                except ValueError:
                    pass
    if found_chg:
        return total

    # Legacy column: 0 = neutral, 1..7 map to +3..-3.
    legacy = {0: 0, 1: 3, 2: 2, 3: 1, 4: 0, 5: -1, 6: -2, 7: -3}
    for line in lines[4 : 4 + n_atoms]:
        try:
            total += legacy.get(int(line[36:39]), 0)
        except (ValueError, IndexError):
            continue
    return total


def _parse_v3000(lines: list[str], name: str, index: int) -> Optional[LigandRecord]:
    n_atoms = 0
    hydrogens = 0
    all_z_zero = True
    charge = 0
    in_atom_block = False

    for line in lines:
        if "BEGIN ATOM" in line:
            in_atom_block = True
            continue
        if "END ATOM" in line:
            in_atom_block = False
            continue
        if "COUNTS" in line:
            parts = line.split()
            if len(parts) > 3:
                try:
                    n_atoms = int(parts[3])
                except ValueError:
                    pass
            continue
        if not in_atom_block or "V30" not in line:
            continue

        parts = line.split()
        # M  V30 <index> <symbol> <x> <y> <z> <aamap> [CHG=n ...]
        if len(parts) < 7:
            continue
        if parts[3] == "H":
            hydrogens += 1
        try:
            if abs(float(parts[6])) > 1e-6:
                all_z_zero = False
        except ValueError:
            pass
        for token in parts[7:]:
            if token.startswith("CHG="):
                try:
                    charge += int(token.split("=", 1)[1])
                except ValueError:
                    pass

    if n_atoms <= 0:
        return None
    return LigandRecord(
        name=name,
        n_atoms=n_atoms,
        n_hydrogens=hydrogens,
        has_3d=not all_z_zero,
        charge=charge,
        index=index,
    )
