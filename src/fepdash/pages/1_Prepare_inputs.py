"""Preparation page -- what OpenFE and TMD need, and whether your files have it.

Deliberately first in the page order. Almost every campaign that dies does
so because of its inputs, and it dies hours in, after the charge-generation
step has already burned CPU. Thirty seconds here is worth a day there.

The checks are plain-text (see core/prep_check.py) because the dashboard's
env has no RDKit. The page says so rather than implying the green ticks mean
the chemistry is right.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fepdash.core import prep_check  # noqa: E402
from fepdash.core.config import load_config  # noqa: E402
from fepdash.core.prep_check import Severity  # noqa: E402
from fepdash.ui.common import page_header  # noqa: E402

st.set_page_config(page_title="Prepare inputs — FEP dashboard", page_icon="🧪", layout="wide")

cfg = load_config()

page_header(
    "Preparing inputs",
    "What a free energy calculation needs from your protein and ligands.",
)

st.info(
    "**The one-line version.** A protein with deliberate protonation and "
    "nothing in it but protein; ligands that are one congeneric series, "
    "posed in the pocket, with explicit hydrogens and the same net charge."
)


# ---------------------------------------------------------------------------
# The checker
# ---------------------------------------------------------------------------

st.subheader("Check your files")
st.caption(
    "Structural checks only — these read the files as text. They catch the "
    "mistakes that kill a campaign in its first minutes. They **cannot** tell "
    "you whether a tautomer, a protonation state, or a perceived bond order "
    "is chemically right; nothing here substitutes for looking at the "
    "structures in Maestro or PyMOL."
)

col_p, col_l = st.columns(2)
with col_p:
    protein_path = st.text_input(
        "Protein (PDB)", value="", placeholder=str(cfg.inputs_root / "protein.pdb")
    )
with col_l:
    ligands_path = st.text_input(
        "Ligands (SDF)", value="", placeholder=str(cfg.inputs_root / "ligands.sdf")
    )

_ICON = {Severity.ERROR: "🔴", Severity.WARN: "🟠", Severity.INFO: "🔵"}


def _render(report) -> None:
    counts = (
        f"{len(report.errors)} error(s), {len(report.warnings)} warning(s)"
        if report.findings
        else "no findings"
    )
    if report.ok and not report.warnings:
        st.success(f"`{report.path.name}` — looks usable ({counts}).")
    elif report.ok:
        st.warning(f"`{report.path.name}` — usable, but check the warnings ({counts}).")
    else:
        st.error(f"`{report.path.name}` — would likely fail ({counts}).")

    if report.facts:
        st.caption(
            " · ".join(f"**{k}**: {v}" for k, v in report.facts.items())
        )

    order = {Severity.ERROR: 0, Severity.WARN: 1, Severity.INFO: 2}
    for finding in sorted(report.findings, key=lambda f: order[f.severity]):
        with st.expander(f"{_ICON[finding.severity]} {finding.title}"):
            if finding.detail:
                st.markdown(finding.detail)
            if finding.fix:
                st.markdown(f"**Fix:** {finding.fix}")


if st.button("Check", type="primary"):
    ran = False
    for label, raw, checker in (
        ("protein", protein_path, prep_check.check_protein),
        ("ligands", ligands_path, prep_check.check_ligands),
    ):
        if not raw.strip():
            continue
        ran = True
        path = Path(raw).expanduser()
        st.markdown(f"#### {label.title()}")
        if not path.is_file():
            st.error(f"Not a file on this box: `{path}`")
            continue
        _render(checker(path))
    if not ran:
        st.caption("Give at least one path above.")


st.divider()

# ---------------------------------------------------------------------------
# Guidance
# ---------------------------------------------------------------------------

tab_protein, tab_ligands, tab_why = st.tabs(
    ["Protein", "Ligands", "Why edges fail"]
)

with tab_protein:
    st.markdown(
        """
### What the protein file has to be

A PDB (or mmCIF) containing **only what you want simulated**. The default
forcefield set is:

`amber/ff14SB` · `tip3p` · `tip3p_HFE_multivalent` · `phosaa10` · `lipid17`

which covers standard amino acids, water, common ions, and phosphorylated
residues. Anything else in the file has no parameters and setup fails.

**Do**

- **Protonate deliberately.** If there are no hydrogens, OpenMM adds them
  using its own defaults — which silently picks your His tautomers and the
  charge on Asp/Glu/Lys. For any residue in the pocket that is a scientific
  decision, not a formatting one. Use PrepWizard, PDB2PQR, or pdb4amber at
  the pH you care about.
- **Remove the ligand.** The molecule you are perturbing comes in through
  the SDF. A copy left in the PDB is an unparameterisable residue *and* a
  steric clash.
- **Pick one altLoc.** Alternate conformations are two coordinate sets for
  the same atoms; keep A (or whichever is right) and drop the rest.
- **Decide about waters.** The protocol solvates the box itself with 1.5 nm
  of TIP3P padding, so crystallographic waters are usually stripped. Keep
  one only if you know it bridges the ligand to the protein.
- **Trim to the chains that matter.** Every chain is simulated. A second
  copy of the protein doubles the cost for nothing unless the site sits at
  the interface.
- **Fix chain breaks near the site.** Unresolved loops leave charged termini
  in the middle of your protein. Model them, or cap them. A gap far from the
  pocket is usually tolerable.

**Cofactors** — a heme, an ATP, a structural metal complex — do not belong
in the protein PDB. Pass them as a cofactor SDF on the Launch page so they
get small-molecule parameters.

**Metals** are the usual sharp edge: a bare ion is fine, but a coordinated
metal centre is not well described by a fixed-charge forcefield, and no
amount of preparation makes that go away.
"""
    )

with tab_ligands:
    st.markdown(
        """
### What the ligand file has to be

One SDF containing **one congeneric series**, every molecule posed in the
binding site, in the same coordinate frame as the protein.

**Do**

- **Pose them in the pocket.** This is the requirement people miss. RBFE
  perturbs one ligand into another *in place* — it does not dock. Every
  ligand needs binding-mode coordinates: aligned to a crystallographic pose,
  or docked and minimised. 3D coordinates from a conformer generator that
  are not in the site are useless.
- **Explicit hydrogens**, at the protonation state you intend. OpenFF does
  not infer them.
- **Correct bond orders and formal charges.** SDF carries these explicitly,
  and a mis-perceived group (sulfonamides and nitro groups are the classic
  offenders when a file has been round-tripped through a format without bond
  orders, like PDB) produces a silently wrong molecule.
- **Unique, meaningful titles.** Both engines key edges and results by
  molecule name. Duplicates collide; blanks make the results table useless.
- **Keep net charge constant.** See below — this is the one that quietly
  produces wrong numbers rather than an error.
- **Keep the series tight.** A shared scaffold with varying substituents.
  Very different ligands map poorly, and a mapping with many dummy atoms
  converges slowly or not at all (TMD caps this at 30 dummy atoms per
  transformation).

### Net charge, specifically

`openfe`'s `explicit_charge_correction` is **off by default**. A
transformation that changes net charge will still run, and still produce
numbers — they just are not comparable to the rest of the network, because
the charge imbalance is absorbed by the solvent box rather than corrected.

So either split the set into same-charge sub-series, or turn on explicit
charge correction (the co-alchemical ion approach) before trusting any edge
that crosses a charge boundary. The checker above flags mixed charges for
this reason.
"""
    )

with tab_why:
    st.markdown(
        """
### The usual causes of a dead campaign

| Symptom | Usual cause |
|---|---|
| Planning fails immediately | Unparameterisable residue in the PDB — a ligand, cofactor, or modified residue left in |
| Planning takes forever | Partial-charge generation is CPU-bound and runs over every ligand; that is normal. Raise CPU cores on the Launch page |
| An edge fails with `Particle coordinate is NaN` | Clashing geometry — ligand not properly posed, or a bad mapping |
| Legs run but ΔΔG is nonsense | Ligands not in the binding site, or a charge-changing edge without correction |
| Wildly inconsistent network | Bad atom mapping on one edge. The Results page's cycle-closure panel localises it to a loop |
| `openfe gather` reports no per-ligand ΔG | Fewer than 2 protocol repeats per edge — set on the Launch page |

### A reasonable prep sequence

1. Get a structure with the ligand bound, or dock into an apo/holo site.
2. Clean the protein: one chain set, one altLoc, no ligand, waters decided.
3. Protonate at your pH; check His tautomers in the pocket by eye.
4. Build the series in the pose frame: align on the common scaffold.
5. Add hydrogens, verify bond orders and formal charges.
6. Write one SDF, unique titles, check net charges match.
7. **Run the checker above**, then launch.

Steps 3 and 5 are the ones worth a human looking at the structures. Nothing
on this page can do them for you.
"""
    )

st.caption(
    "These notes describe the default settings this dashboard launches with. "
    "If you change the mapper, network planner, or charge method on the "
    "Launch page, some of the above shifts with it."
)
