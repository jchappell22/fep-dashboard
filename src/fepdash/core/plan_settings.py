"""fepdash.core.plan_settings -- build openfe's planning YAML from form values.

Why the dashboard writes this file instead of asking for one
-----------------------------------------------------------

``openfe plan-rbfe-network -s settings.yaml`` is how you choose the atom
mapper, the network planner, and the partial-charge method. Making a
scientist hand-write that YAML means memorising registry key spellings and
kwarg names, with no feedback until planning dies -- after the expensive
charge-generation step.

So the Launch page renders widgets instead, and this module turns their
values into the file. The *options* are declared in ``engines/openfe.toml``
(``[rbfe.settings]``), not here: adding a network planner openfe gained last
week is a TOML edit, exactly like every other engine command in this repo.

The one subtlety that matters
-----------------------------

openfe applies a supplied mapper section as ``cls(**settings)``, so any key
you omit falls back to the *class* default. But openfe's own no-YAML path
constructs Kartograf with ``map_hydrogens_on_hydrogens_only=True``, which
its source comments call a "non-default setting".

Net effect: supplying a mapper section and omitting that key silently
changes mapping behaviour versus running plain ``openfe``. So we emit every
non-optional field explicitly rather than trying to be clever about which
ones "look default". Only fields marked ``optional`` are dropped when blank,
because for those an omitted key is the intended way to say "let openfe
decide" -- writing a literal ``None`` would set the string "None".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import yaml


@dataclass
class Choice:
    """One selectable method within a section."""

    value: str  # the literal lowercase registry key openfe looks up
    label: str
    help: str = ""
    available: bool = True


@dataclass
class Field:
    """One tunable setting, belonging to one or more methods."""

    name: str
    type: str  # float | int | bool | str | choice
    default: Any
    applies_to: tuple[str, ...] = ()
    help: str = ""
    options: tuple[str, ...] = ()  # for type == "choice"
    optional: bool = False  # blank/zero means "omit the key entirely"
    required: bool = False  # blank is a launch-blocking error
    step: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None


@dataclass
class Section:
    """A YAML top-level key: mapper, network, or partial_charge."""

    key: str
    label: str
    default: str
    help: str = ""
    choices: list[Choice] = field(default_factory=list)
    fields: list[Field] = field(default_factory=list)

    def fields_for(self, method: str) -> list[Field]:
        return [f for f in self.fields if method in f.applies_to]

    def choice(self, value: str) -> Optional[Choice]:
        return next((c for c in self.choices if c.value == value), None)


# ---------------------------------------------------------------------------
# Loading the schema out of the engine TOML
# ---------------------------------------------------------------------------


def load_schema(engine, method) -> dict[str, Section]:
    """Read ``[<method>.settings]`` from an engine definition.

    Returns ``{}`` when the engine declares none -- which is the correct
    answer for TMD (its launcher owns its own defaults) and for OpenFE ABFE
    (no planner involved).
    """
    # `[<method>.settings]` is a sibling of `[<method>.plan]`, not nested
    # inside it -- settings describe the method, not one stage of it.
    spec = (engine.method_raw.get(method.value) or {}).get("settings")
    if not isinstance(spec, dict):
        return {}

    sections: dict[str, Section] = {}
    for key, body in spec.items():
        if not isinstance(body, dict):
            continue
        sections[key] = Section(
            key=key,
            label=str(body.get("label", key)),
            default=str(body.get("default", "")),
            help=str(body.get("help", "")),
            choices=[
                Choice(
                    value=str(c["value"]),
                    label=str(c.get("label", c["value"])),
                    help=str(c.get("help", "")),
                    available=bool(c.get("available", True)),
                )
                for c in body.get("choices", [])
                if "value" in c
            ],
            fields=[
                Field(
                    name=str(f["name"]),
                    type=str(f.get("type", "str")),
                    default=f.get("default"),
                    applies_to=tuple(str(m) for m in f.get("applies_to", [])),
                    help=str(f.get("help", "")),
                    options=tuple(str(o) for o in f.get("options", [])),
                    optional=bool(f.get("optional", False)),
                    required=bool(f.get("required", False)),
                    step=f.get("step"),
                    min=f.get("min"),
                    max=f.get("max"),
                )
                for f in body.get("fields", [])
                if "name" in f
            ],
        )
    return sections


# ---------------------------------------------------------------------------
# Form values -> YAML
# ---------------------------------------------------------------------------


def _omit(value: Any, spec: Field) -> bool:
    """Should this field be left out of the YAML entirely?

    Only ever true for ``optional`` fields. A blank optional string or a
    zero/negative optional number means "no opinion, let openfe decide", and
    the way to express that is an absent key -- not ``null``, and certainly
    not the string ``"None"`` (which openfe's own DEFAULT_YAML docstring
    shows, and which would be set verbatim by its ``setattr`` loop).
    """
    if not spec.optional:
        return False
    if spec.type in ("str", "choice"):
        return not str(value).strip()
    if spec.type in ("int", "float"):
        try:
            return float(value) <= 0
        except (TypeError, ValueError):
            return True
    return False


def _coerce(value: Any, spec: Field) -> Any:
    if spec.type == "bool":
        return bool(value)
    if spec.type == "int":
        return int(value)
    if spec.type == "float":
        return float(value)
    return str(value)


def build_settings(
    schema: dict[str, Section], values: dict[str, Any]
) -> dict[str, Any]:
    """Assemble the nested dict openfe's parser expects.

    ``values`` is flat and namespaced: ``{"mapper": "kartograf",
    "mapper.atom_max_distance": 0.95, ...}``.
    """
    out: dict[str, Any] = {}
    for key, section in schema.items():
        method = str(values.get(key, section.default) or section.default)
        if not method:
            continue
        settings: dict[str, Any] = {}
        for spec in section.fields_for(method):
            raw = values.get(f"{key}.{spec.name}", spec.default)
            if _omit(raw, spec):
                continue
            settings[spec.name] = _coerce(raw, spec)

        entry: dict[str, Any] = {"method": method}
        if settings:
            entry["settings"] = settings
        out[key] = entry
    return out


def build_yaml(schema: dict[str, Section], values: dict[str, Any]) -> str:
    """Render the settings file, with a header explaining where it came from."""
    data = build_settings(schema, values)
    if not data:
        return ""
    body = yaml.safe_dump(data, sort_keys=False, default_flow_style=False)
    return (
        "# Generated by fep-dashboard from the Launch page -- do not hand-edit.\n"
        "# Consumed by `openfe plan-rbfe-network -s`. Kept alongside the\n"
        "# campaign as the record of how its network was planned.\n"
        "#\n"
        "# Every non-optional key is written explicitly, including ones at\n"
        "# their default: openfe applies a mapper section as cls(**settings),\n"
        "# so an omitted key silently reverts to the class default rather\n"
        "# than the CLI default openfe would otherwise have used.\n"
        "\n" + body
    )


def validate(schema: dict[str, Section], values: dict[str, Any]) -> list[str]:
    """Problems that would make planning fail. Empty list means go.

    The one that matters is ``generate_radial_network``'s ``central_ligand``:
    it has no default, so leaving it blank kills planning *after* the
    charge-generation step -- the slow part. Catching it on the form is the
    difference between a corrected typo and a wasted hour.
    """
    problems: list[str] = []
    for key, section in schema.items():
        method = str(values.get(key, section.default) or section.default)
        choice = section.choice(method)
        if choice is None:
            problems.append(
                f"{section.label}: '{method}' is not one of "
                f"{[c.value for c in section.choices]}"
            )
            continue
        if not choice.available:
            problems.append(
                f"{section.label}: '{choice.label}' is not usable on this box. "
                f"{choice.help}"
            )
        for spec in section.fields_for(method):
            if not spec.required:
                continue
            raw = values.get(f"{key}.{spec.name}", spec.default)
            if not str(raw).strip():
                problems.append(
                    f"{section.label} → {spec.name} is required for "
                    f"'{choice.label}' and has no default. {spec.help}"
                )
    return problems


def summarise(schema: dict[str, Section], values: dict[str, Any]) -> str:
    """One-line human summary, for the campaign row and the run header."""
    bits = []
    for key, section in schema.items():
        method = str(values.get(key, section.default) or section.default)
        choice = section.choice(method)
        bits.append(f"{section.label}: {choice.label if choice else method}")
    return " · ".join(bits)
