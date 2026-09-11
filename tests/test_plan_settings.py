"""Tests for the generated openfe planning-settings YAML.

The important test here is :func:`test_every_combination_loads_in_openfe`.
It does not check that we emit *some* YAML -- it feeds every combination the
engine TOML declares to **openfe's own parser and object resolver**, which
is what actually catches a misspelled registry key or a kwarg that no longer
exists. That makes this the strongest guarantee in the repo: the feature is
verified against the real library rather than against a reading of it.

It can only run where openfe is importable, so it skips in the `fep-dash`
env and runs in an OpenFE env:

    /opt/openfe/current/bin/python -m pytest tests/test_plan_settings.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fepdash.core import plan_settings  # noqa: E402
from fepdash.core.engines.base import load_engines  # noqa: E402
from fepdash.core.models import Method  # noqa: E402

ENGINES_DIR = Path(__file__).resolve().parents[1] / "engines"



@pytest.fixture(scope="module")
def schema():
    engine = load_engines(ENGINES_DIR)["openfe"]
    return plan_settings.load_schema(engine, Method.RBFE)


def _defaults(schema) -> dict:
    """Form values as the Launch page would first render them."""
    values = {}
    for key, section in schema.items():
        values[key] = section.default
        for spec in section.fields_for(section.default):
            values[f"{key}.{spec.name}"] = spec.default
    return values


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_schema_declares_all_three_sections(schema):
    """mapper / network / partial_charge is the complete surface openfe's
    parser recognises -- anything else it warns about and ignores."""
    assert set(schema) == {"mapper", "network", "partial_charge"}


def test_tmd_declares_no_settings():
    """TMD's launcher owns its own defaults; offering a settings form for it
    would imply control the dashboard does not have."""
    engine = load_engines(ENGINES_DIR)["tmd"]
    assert plan_settings.load_schema(engine, Method.RBFE) == {}


def test_openfe_abfe_declares_no_settings():
    """No network planner is involved, and the ABFE shim ignores -s."""
    engine = load_engines(ENGINES_DIR)["openfe"]
    assert plan_settings.load_schema(engine, Method.ABFE) == {}


# ---------------------------------------------------------------------------
# Emission rules
# ---------------------------------------------------------------------------


def test_kartograf_hydrogen_flag_is_always_emitted(schema):
    """The trap this whole module exists to avoid.

    openfe applies a mapper section as ``cls(**settings)``, so an omitted key
    falls back to the CLASS default (False). But openfe's own no-YAML path
    sets map_hydrogens_on_hydrogens_only=True. Omitting it because it "looks
    default" would silently change mapping behaviour versus plain openfe.
    """
    data = plan_settings.build_settings(schema, _defaults(schema))
    assert data["mapper"]["settings"]["map_hydrogens_on_hydrogens_only"] is True


def test_blank_optional_fields_are_omitted_not_nulled(schema):
    """openfe setattr's whatever we write. Emitting `None` would set the
    string "None" -- which is exactly what openfe's own DEFAULT_YAML
    docstring mistakenly shows."""
    values = _defaults(schema)
    values["mapper"] = "lomap"
    values["mapper.seed"] = ""
    data = plan_settings.build_settings(schema, values)
    assert "seed" not in data["mapper"]["settings"]

    text = plan_settings.build_yaml(schema, values)
    assert "None" not in text and "null" not in text


def test_zero_optional_number_is_omitted(schema):
    values = _defaults(schema)
    values["partial_charge.number_of_conformers"] = 0
    data = plan_settings.build_settings(schema, values)
    assert "number_of_conformers" not in data["partial_charge"]["settings"]


def test_booleans_emit_as_yaml_booleans(schema):
    text = plan_settings.build_yaml(schema, _defaults(schema))
    assert "true" in text  # not Python's "True"
    assert "True" not in text


def test_only_fields_for_the_chosen_method_are_emitted(schema):
    values = _defaults(schema)
    values["mapper"] = "lomap"
    for spec in schema["mapper"].fields_for("lomap"):
        values[f"mapper.{spec.name}"] = spec.default
    data = plan_settings.build_settings(schema, values)
    emitted = set(data["mapper"]["settings"])
    assert "time" in emitted  # a lomap field
    assert "atom_max_distance" not in emitted  # a kartograf field


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_radial_without_central_ligand_blocks(schema):
    """central_ligand has no default; planning would die AFTER the expensive
    charge-generation step."""
    values = _defaults(schema)
    values["network"] = "generate_radial_network"
    values["network.central_ligand"] = ""
    problems = plan_settings.validate(schema, values)
    assert any("central_ligand" in p for p in problems)


def test_radial_with_central_ligand_passes(schema):
    values = _defaults(schema)
    values["network"] = "generate_radial_network"
    values["network.central_ligand"] = "lig_ejm_31"
    assert plan_settings.validate(schema, values) == []


def test_unavailable_charge_method_is_flagged(schema):
    """OpenEye is not licensed on this box; am1bccelf10 would just fail."""
    values = _defaults(schema)
    values["partial_charge"] = "am1bccelf10"
    problems = plan_settings.validate(schema, values)
    assert any("am1bcc" in p.lower() or "OpenEye" in p for p in problems)


def test_defaults_validate_clean(schema):
    assert plan_settings.validate(schema, _defaults(schema)) == []


# ---------------------------------------------------------------------------
# The real test: does openfe itself accept what we generate?
# ---------------------------------------------------------------------------


def _openfe_loader():
    """openfe's own parser + object resolver, or None if openfe is absent."""
    try:
        from openfecli.parameters.plan_network_options import (
            load_yaml_planner_options,
            parse_yaml_planner_options,
        )
    except ImportError:
        return None
    return parse_yaml_planner_options, load_yaml_planner_options


@pytest.mark.parametrize("mapper", ["kartograf", "lomap"])
@pytest.mark.parametrize(
    "network",
    [
        "generate_minimal_spanning_network",
        "generate_minimal_redundant_network",
        "generate_lomap_network",
        "generate_radial_network",
        "generate_maximal_network",
    ],
)
def test_every_combination_loads_in_openfe(schema, tmp_path, mapper, network):
    """Feed generated YAML to openfe's real resolver.

    ``load_yaml_planner_options`` performs the registry lookups, constructs
    the mapper as ``cls(**settings)``, and builds ``partial(func, **settings)``
    for the network planner. So a bad method name or a kwarg the function
    does not accept fails HERE -- which is the whole point.
    """
    loader = _openfe_loader()
    if loader is None:
        pytest.skip("openfe not importable in this env")
    parse, load = loader

    values = _defaults(schema)
    values["mapper"] = mapper
    for spec in schema["mapper"].fields_for(mapper):
        values[f"mapper.{spec.name}"] = spec.default
    values["network"] = network
    for spec in schema["network"].fields_for(network):
        values[f"network.{spec.name}"] = spec.default
    # The one field with no default; any name is fine for resolution, since
    # matching it against the SDF happens later, during planning.
    if network == "generate_radial_network":
        values["network.central_ligand"] = "lig_ejm_31"

    text = plan_settings.build_yaml(schema, values)
    parsed = parse(text)  # schema-level validation
    assert parsed is not None

    path = tmp_path / "plan_settings.yaml"
    path.write_text(text)
    options = load(str(path), None)  # registry + kwargs validation

    assert options.mapper is not None
    assert options.ligand_network_planner is not None
    assert options.partial_charge.partial_charge_method == "am1bcc"


def test_generated_yaml_is_valid_yaml(schema):
    text = plan_settings.build_yaml(schema, _defaults(schema))
    data = yaml.safe_load(text)
    assert set(data) == {"mapper", "network", "partial_charge"}
    assert data["mapper"]["method"] == "kartograf"
