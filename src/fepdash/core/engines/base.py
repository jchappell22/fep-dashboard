"""fepdash.core.engines.base -- load engine TOMLs and render their commands.

The whole point of this module: **no openfe or tmd flag appears in Python.**
An engine is a TOML file describing three stages (plan / run / gather) as
command templates. This module loads them, validates them, and renders them
against a campaign.

Why templates instead of code
-----------------------------

The dashboard is developed on a laptop and runs on GPU2, where the actual
engine versions live. Jacob's own openfe scripts carry the comment *"your
openfe version's flags may differ"* -- that is the normal case, not an edge
case. With templates, a flag mismatch is a one-line edit to a TOML on the
box. With hardcoded argv, it is a code change made blind.

Rendering rules
---------------

* ``{placeholder}`` substitution only -- no expressions, no conditionals.
* An unknown placeholder raises :class:`TemplateError`. It never renders as
  an empty string, because a silently-dropped ``--pdb_path`` is a 12-hour
  run that produces garbage.
* Optional flags are handled by dedicated ``{opt_*}`` variables that the
  adapter fills with either ``""`` or ``" -C /path"`` -- leading space
  included. This keeps the conditional in Python where it can be tested,
  and the flag spelling in TOML where it can be fixed.
"""

from __future__ import annotations

import shlex
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

from ..models import Campaign, Method


class TemplateError(ValueError):
    """A command template referenced a variable nobody supplied."""


class EngineError(ValueError):
    """An engine TOML is missing something the driver needs."""


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render(template: str, variables: dict[str, Any], *, where: str) -> str:
    """Substitute ``{name}`` placeholders; collapse the line continuations.

    ``where`` is only used to make the error message name the offending
    TOML key, e.g. ``openfe/rbfe.run.cmd``.
    """
    if not template or not template.strip():
        return ""

    required = {
        fname
        for _, fname, _, _ in string.Formatter().parse(template)
        if fname
    }
    missing = sorted(required - set(variables))
    if missing:
        raise TemplateError(
            f"{where}: template needs {missing} but the campaign supplied "
            f"{sorted(variables)}. Either add the value on the Launch page "
            f"or fix the placeholder in the engine TOML."
        )

    rendered = template.format(**{k: _stringify(v) for k, v in variables.items()})
    # TOML multi-line strings keep the trailing backslashes we wrote for
    # readability; bash is happy with them, but collapsing makes the command
    # preview on the Launch page one legible line.
    return " ".join(rendered.split())


def _campaign_path(subdir: str) -> str:
    """A campaign subdirectory as a quoted, ``$CAMPAIGN_DIR``-relative bash word.

    Returned as a pre-quoted string rather than a Path so :func:`_stringify`
    leaves it alone -- shlex-quoting it would escape the ``$`` and turn the
    variable reference into a literal.
    """
    return f'"$CAMPAIGN_DIR/{subdir}"'


def _stringify(value: Any) -> str:
    if isinstance(value, Path):
        return shlex.quote(str(value))
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


# ---------------------------------------------------------------------------
# Engine definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageSpec:
    """One stage of one method, straight out of the TOML."""

    cmds: tuple[str, ...]
    raw: dict[str, Any]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)


@dataclass(frozen=True)
class Engine:
    """A loaded ``engines/<name>.toml``."""

    name: str
    label: str
    methods: tuple[str, ...]
    env: dict[str, Any]
    defaults: dict[str, Any]
    stages: dict[str, dict[str, StageSpec]]  # method -> stage -> spec
    source: Path
    #: "explicit" -- the dashboard picks cards and pins them.
    #: "auto"     -- the engine's own launcher picks; the dashboard must not
    #:               interfere, must not export CUDA_VISIBLE_DEVICES, and must
    #:               not claim cards it cannot actually reserve.
    gpu_selection: str = "explicit"

    # -- accessors --------------------------------------------------------

    def supports(self, method: Method) -> bool:
        return method.value in self.methods

    def stage(self, method: Method, stage: str) -> StageSpec:
        try:
            return self.stages[method.value][stage]
        except KeyError:
            raise EngineError(
                f"engine '{self.name}' has no [{method.value}.{stage}] section "
                f"in {self.source}"
            ) from None

    @property
    def is_single_process(self) -> bool:
        """True when the engine schedules its own GPU work (TMD via MPS).

        Drives which driver template we render: a fan-out loop over
        ``CUDA_VISIBLE_DEVICES``, or one process handed the whole GPU set.
        """
        for method_stages in self.stages.values():
            run = method_stages.get("run")
            if run is not None and run.get("parallel") == "single":
                return True
        return False

    def parallel_mode(self, method: Method) -> str:
        return self.stage(method, "run").get("parallel", "per_gpu")

    @property
    def picks_own_gpu(self) -> bool:
        """True when the engine's launcher chooses cards for itself.

        TMD's ``tmd-submit`` auto-picks the most-free GPU on the shared box.
        Overriding that with ``CUDA_VISIBLE_DEVICES`` would defeat the
        sharing behaviour everyone else on the box depends on, so the
        dashboard stays out of the way and the Launch page hides its GPU
        picker rather than offering a control that does nothing.
        """
        return self.gpu_selection == "auto"

    @property
    def conda_env(self) -> Optional[str]:
        """None when the engine is reached through a launcher on PATH."""
        return self.env.get("conda_env")

    @property
    def conda_root(self) -> Optional[str]:
        return self.env.get("conda_root")

    @property
    def needs_conda(self) -> bool:
        """Whether the driver must activate an env before calling the binary.

        Both engines on Conifer are reached through ``/usr/local/bin``
        launchers that activate their own shared env internally, so this is
        normally False. It stays supported for an engine added later that is
        only available inside a personal env.
        """
        return bool(self.env.get("conda_env"))

    @property
    def version_spec(self) -> dict[str, Any]:
        """``[env.versions]``: how to ask the launcher for a specific build."""
        return self.env.get("versions", {}) or {}

    @property
    def available_versions(self) -> list[str]:
        return [str(v) for v in self.version_spec.get("available", [])]

    @property
    def version_env_var(self) -> Optional[str]:
        var = self.version_spec.get("env_var")
        return str(var) if var else None

    @property
    def binary(self) -> str:
        return self.env.get("binary", "")

    @property
    def engine_workdir(self) -> Optional[str]:
        """Directory the engine's commands must run from (TMD's checkout)."""
        return self.env.get("workdir")

    @property
    def exports(self) -> dict[str, str]:
        raw = self.env.get("exports", {})
        return {str(k): str(v) for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

_STAGES = ("plan", "run", "gather")


def load_engine(path: Path) -> Engine:
    """Parse one engine TOML. Raises :class:`EngineError` on a bad file."""
    with path.open("rb") as fh:
        raw = tomllib.load(fh)

    name = raw.get("name") or path.stem
    methods = tuple(raw.get("methods") or [])
    if not methods:
        raise EngineError(f"{path}: no `methods` declared")

    stages: dict[str, dict[str, StageSpec]] = {}
    for method in methods:
        section = raw.get(method)
        if not isinstance(section, dict):
            raise EngineError(f"{path}: declares method '{method}' but has no [{method}] table")
        per_stage: dict[str, StageSpec] = {}
        for stage in _STAGES:
            spec = section.get(stage)
            if not isinstance(spec, dict):
                raise EngineError(f"{path}: missing [{method}.{stage}]")
            # A stage is either one `cmd` or a list of `cmds`; normalise.
            if "cmds" in spec:
                cmds = tuple(str(c) for c in spec["cmds"])
            else:
                one = str(spec.get("cmd", "") or "")
                cmds = (one,) if one.strip() else ()
            per_stage[stage] = StageSpec(cmds=cmds, raw=spec)
        stages[method] = per_stage

    return Engine(
        name=str(name),
        label=str(raw.get("label", name)),
        methods=methods,
        env=raw.get("env", {}) or {},
        defaults=raw.get("defaults", {}) or {},
        stages=stages,
        source=path,
        gpu_selection=str(raw.get("gpu_selection", "explicit")),
    )


def load_engines(engines_dir: Path) -> dict[str, Engine]:
    """Load every ``*.toml`` in ``engines_dir``, keyed by engine name.

    A malformed file is skipped with its error attached rather than taking
    the whole dashboard down -- a typo in tmd.toml should not stop you
    looking at OpenFE results.
    """
    found: dict[str, Engine] = {}
    if not engines_dir.is_dir():
        return found
    for path in sorted(engines_dir.glob("*.toml")):
        try:
            engine = load_engine(path)
        except (EngineError, tomllib.TOMLDecodeError) as exc:  # type: ignore[attr-defined]
            found[path.stem] = _broken(path, exc)
            continue
        found[engine.name] = engine
    return found


def _broken(path: Path, exc: Exception) -> Engine:
    """A placeholder engine that reports its own parse failure in the UI."""
    return Engine(
        name=path.stem,
        label=f"{path.stem} -- BROKEN: {exc}",
        methods=(),
        env={},
        defaults={},
        stages={},
        source=path,
    )


# ---------------------------------------------------------------------------
# Campaign -> render variables
# ---------------------------------------------------------------------------


def build_variables(
    campaign: Campaign,
    engine: Engine,
    *,
    python: str = "python",
    scripts_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Everything a template is allowed to reference.

    Engine params from the launch form are merged in last, so a campaign can
    override an engine default without the TOML knowing the key exists.
    """
    variables: dict[str, Any] = {
        # Inputs stay absolute -- they live outside the campaign.
        "ligands": campaign.ligands,
        "protein": campaign.protein,
        # Campaign directories are expressed RELATIVE to $CAMPAIGN_DIR, which
        # driver.sh resolves from its own location. That makes the generated
        # script relocatable: a campaign directory can be moved, copied to
        # another box, or archived and re-run, and the script still works.
        # Baking in the absolute path at generation time would silently
        # break all three.
        "plan_dir": _campaign_path("plan"),
        "work_dir": _campaign_path("work"),
        "results_dir": _campaign_path("results"),
        "gathered_dir": _campaign_path("gathered"),
        "logs_dir": _campaign_path("legs"),
        "run_dir": '"$CAMPAIGN_DIR"',
        # interpreters / locations
        "python": python,
        "scripts_dir": scripts_dir or "",
        "engine_workdir": engine.engine_workdir or "",
        # optional flags -- rendered here so the conditional is testable and
        # the flag spelling stays in the TOML.
        "opt_cofactors": f" -C {shlex.quote(str(campaign.cofactors))}" if campaign.cofactors else "",
        "opt_settings": f" -s {shlex.quote(str(campaign.settings_yaml))}" if campaign.settings_yaml else "",
        "opt_resume": "",
        # Per-leg placeholders: the driver script substitutes these in bash,
        # so they must survive Python rendering untouched. Quoted, because a
        # transformation filename or campaign path containing a space would
        # otherwise word-split into two broken arguments.
        "job": '"$tf"',
        "leg": '"$name"',
    }
    variables.update(engine.defaults)
    variables.update(campaign.params)
    return variables


def preview_commands(
    campaign: Campaign,
    engine: Engine,
    *,
    python: str = "python",
    scripts_dir: Optional[Path] = None,
) -> dict[str, list[str]]:
    """Render every stage for display on the Launch page before committing.

    This is the cheap way to catch a bad template: you see the literal
    command line you are about to run for 12 hours.
    """
    variables = build_variables(campaign, engine, python=python, scripts_dir=scripts_dir)
    out: dict[str, list[str]] = {}
    for stage in _STAGES:
        spec = engine.stage(campaign.method, stage)
        rendered = []
        for i, tmpl in enumerate(spec.cmds):
            where = f"{engine.name}/{campaign.method.value}.{stage}.cmd[{i}]"
            text = render(tmpl, variables, where=where)
            if text:
                rendered.append(text)
        out[stage] = rendered
    return out
