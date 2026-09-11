#!/usr/bin/env python
"""Build OpenFE ABFE transformation JSONs -- one per ligand.

Why this exists
---------------

OpenFE's CLI has ``plan-rbfe-network`` but no ``plan-abfe-network``: ABFE
setup goes through the Python layer. This script is that layer, wrapped in
a CLI so the dashboard can drive it the same way it drives everything else
-- as a command template in ``engines/openfe.toml``.

It must run INSIDE an OpenFE env, which the shared ``/opt`` launcher does
not give us for an arbitrary script. So invoke it with that env's python:

    /opt/openfe/current/bin/python scripts/openfe_setup_abfe.py --help

Lambda schedule
---------------

The default schedule is not OpenFE's. It carries a fix from a real campaign
on this box (ST4_FEP): restraint introduction was the entire uncertainty
budget, with the 0 -> 0.2 step at sigma = 3.4 kT (~2 kcal/mol) while every
other transition was under 0.08 kT. The default below bridges 0 -> 0.2 with
intermediate states, which is what made that campaign's error budget
tolerable. ``--default-lambdas`` reverts to OpenFE's stock schedule if you
want to compare.

Output: ``<output-dir>/transformations/<ligand>.json``, which is what
``openfe quickrun`` consumes and what the dashboard's ``jobs_glob`` finds.
"""

from __future__ import annotations

import argparse
import pathlib
import sys


# The schedule from the ST4 campaign. Restraint introduction gets 9 states
# concentrated at the low-lambda end, where the free energy changes fastest.
RESTRAINT_INTRO = [0.0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
ELEC_OFF = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
VDW_OFF = [
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65,
    0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0,
]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--ligands", required=True, type=pathlib.Path,
                   help="SDF containing the ligands to run.")
    p.add_argument("--protein", required=True, type=pathlib.Path,
                   help="Prepared protein PDB.")
    p.add_argument("--output-dir", required=True, type=pathlib.Path,
                   help="Transformations are written to <output-dir>/transformations/.")
    p.add_argument("--n-protocol-repeats", type=int, default=3,
                   help="Repeats run in serial per `openfe quickrun` call (default 3).")
    p.add_argument("--only", default="",
                   help="Comma-separated ligand names; default is every ligand in the SDF.")
    # -C and -s mirror `openfe plan-rbfe-network`'s spelling deliberately:
    # the dashboard renders one shared pair of optional-flag fragments for
    # both methods, so the shim has to accept the same short flags the real
    # CLI uses.
    p.add_argument("-C", "--cofactors", type=pathlib.Path, default=None,
                   help="Optional cofactors SDF, included in both end states.")
    p.add_argument("-s", "--settings", type=pathlib.Path, default=None,
                   help="Accepted and ignored; present so the command template "
                        "stays uniform with the RBFE path.")
    p.add_argument("--default-lambdas", action="store_true",
                   help="Use OpenFE's stock lambda schedule instead of the "
                        "restraint-bridged one documented above.")
    p.add_argument("--equilibration-ns", type=float, default=2.0)
    p.add_argument("--target-error", type=float, default=0.12,
                   help="Early-termination target error, kcal/mol (default 0.12).")
    p.add_argument("--n-processors", type=int, default=1,
                   help="Processors for partial-charge generation.")
    return p.parse_args(argv)


def load_ligands(sdf_path: pathlib.Path, only: set[str]):
    from rdkit import Chem
    import openfe

    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mols, found, skipped = [], set(), 0
    for mol in supplier:
        if mol is None:
            # A molecule RDKit cannot parse is reported, never silently
            # dropped -- a missing ligand three hours in is a bad surprise.
            skipped += 1
            continue
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
        if only and name not in only:
            continue
        if not name:
            raise SystemExit(
                f"a molecule in {sdf_path} has no _Name; ABFE transformations "
                f"are keyed by ligand name, so every record needs one"
            )
        mols.append(openfe.SmallMoleculeComponent.from_rdkit(mol))
        found.add(name)

    if skipped:
        print(f"WARNING: RDKit could not parse {skipped} record(s) in {sdf_path}",
              file=sys.stderr)
    missing = only - found
    if missing:
        raise SystemExit(f"ligand(s) not found in {sdf_path}: {sorted(missing)}")
    if not mols:
        raise SystemExit(f"no usable ligands loaded from {sdf_path}")
    return mols


def apply_lambda_schedule(settings) -> int:
    """Install the restraint-bridged schedule. Returns the replica count."""
    n_states = len(RESTRAINT_INTRO) + len(ELEC_OFF) + len(VDW_OFF)

    settings.complex_lambda_settings.lambda_restraints = (
        RESTRAINT_INTRO + [1.0] * (len(ELEC_OFF) + len(VDW_OFF))
    )
    settings.complex_lambda_settings.lambda_elec = (
        [0.0] * len(RESTRAINT_INTRO) + ELEC_OFF + [1.0] * len(VDW_OFF)
    )
    settings.complex_lambda_settings.lambda_vdw = (
        [0.0] * (len(RESTRAINT_INTRO) + len(ELEC_OFF)) + VDW_OFF
    )
    settings.complex_simulation_settings.n_replicas = n_states
    return n_states


def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        import openfe
        from openff.units import unit
        from openfe.protocols.openmm_afe import AbsoluteBindingProtocol
        from openfe.protocols.openmm_utils.charge_generation import (
            bulk_assign_partial_charges,
        )
        from openfe.protocols.openmm_utils.omm_settings import (
            OpenFFPartialChargeSettings,
        )
    except ImportError as exc:
        raise SystemExit(
            f"cannot import openfe ({exc}). Run this with an OpenFE env's "
            f"python, e.g. /opt/openfe/current/bin/python"
        ) from exc

    only = {n.strip() for n in args.only.split(",") if n.strip()}
    ligands = load_ligands(args.ligands, only)
    print(f"loaded {len(ligands)} ligand(s): {[l.name for l in ligands]}")

    # Charges are generated once, up front, so every transformation uses an
    # identical set. Generating them per-transformation is how you end up
    # comparing ligands that were charged differently.
    charge_settings = OpenFFPartialChargeSettings(
        partial_charge_method="am1bcc",
        off_toolkit_backend="ambertools",
    )
    print("generating am1bcc partial charges (CPU-bound; this is the slow part)...")
    ligands = bulk_assign_partial_charges(
        molecules=ligands,
        overwrite=False,
        method=charge_settings.partial_charge_method,
        toolkit_backend=charge_settings.off_toolkit_backend,
        generate_n_conformers=charge_settings.number_of_conformers,
        nagl_model=charge_settings.nagl_model,
        processors=args.n_processors,
    )

    protein = openfe.ProteinComponent.from_pdb_file(str(args.protein))
    solvent = openfe.SolventComponent()

    extra: dict = {}
    if args.cofactors:
        from rdkit import Chem
        for i, mol in enumerate(Chem.SDMolSupplier(str(args.cofactors), removeHs=False)):
            if mol is None:
                continue
            extra[f"cofactor_{i}"] = openfe.SmallMoleculeComponent.from_rdkit(mol)
        print(f"including {len(extra)} cofactor(s)")

    settings = AbsoluteBindingProtocol.default_settings()
    settings.protocol_repeats = args.n_protocol_repeats

    if args.default_lambdas:
        print("using OpenFE's stock lambda schedule")
    else:
        n_states = apply_lambda_schedule(settings)
        print(f"using the restraint-bridged lambda schedule ({n_states} states)")

    settings.complex_simulation_settings.equilibration_length = (
        args.equilibration_ns * unit.nanosecond
    )
    for phase in ("complex_simulation_settings", "solvent_simulation_settings"):
        getattr(settings, phase).early_termination_target_error = (
            args.target_error * unit.kilocalorie_per_mole
        )

    settings.restraint_settings.host_min_distance = 0.7 * unit.nanometer
    settings.restraint_settings.host_max_distance = 1.1 * unit.nanometer
    settings.engine_settings.compute_platform = "CUDA"

    protocol = AbsoluteBindingProtocol(settings=settings)

    out_dir = args.output_dir / "transformations"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ligand in ligands:
        state_a = openfe.ChemicalSystem(
            {"ligand": ligand, "protein": protein, "solvent": solvent, **extra},
            name=ligand.name,
        )
        # State B is the same system with the ligand removed -- that is what
        # makes this absolute rather than relative.
        state_b = openfe.ChemicalSystem(
            {"protein": protein, "solvent": solvent, **extra}
        )
        transformation = openfe.Transformation(
            stateA=state_a,
            stateB=state_b,
            mapping=None,
            protocol=protocol,
            name=ligand.name,
        )
        out_path = out_dir / f"{ligand.name}.json"
        transformation.dump(out_path)
        print(f"  wrote {out_path}")

    print(f"\n{len(ligands)} transformation(s) in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
