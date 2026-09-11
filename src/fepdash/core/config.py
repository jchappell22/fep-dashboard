"""fepdash.core.config -- where things live on this box.

Resolution order (first hit wins):

1. ``$FEPDASH_CONFIG``  -- explicit path to a config.toml
2. ``<repo>/config.toml``
3. built-in defaults (repo-relative; fine for a laptop, wrong for GPU2)

Everything is a path or a small scalar. Engine *commands* are not here --
they live in ``engines/*.toml``, one file per engine, so that fixing a flag
mismatch on GPU2 never means touching this file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:  # tomllib joined stdlib in 3.11; tomli is the same parser for 3.10.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


@dataclass(frozen=True)
class Config:
    #: Root under which campaign directories are created.
    runs_root: Path
    #: Sqlite file holding the campaigns table.
    db_path: Path
    #: Directory of engine definition TOMLs.
    engines_dir: Path
    #: Directory scanned by the launch form for input structures/ligands.
    inputs_root: Path
    #: Helper scripts shipped with the repo (the ABFE setup shim lives here).
    scripts_dir: Path
    #: GPU indices this dashboard is allowed to schedule onto.
    allowed_gpus: tuple[int, ...]
    #: Refuse to launch when a requested GPU is already claimed. Turning this
    #: off is how you deliberately oversubscribe; see core/gpu.py.
    enforce_gpu_locks: bool = True


def repo_root() -> Path:
    """``<repo>`` -- three parents up from this file (src/fepdash/core)."""
    return Path(__file__).resolve().parents[3]


def _defaults() -> Config:
    root = repo_root()
    return Config(
        runs_root=root / "runs",
        db_path=root / "runs" / "state.db",
        engines_dir=root / "engines",
        inputs_root=root / "inputs",
        scripts_dir=root / "scripts",
        allowed_gpus=(0,),
        enforce_gpu_locks=True,
    )


def _resolve_config_path(explicit: Optional[Path] = None) -> Optional[Path]:
    if explicit is not None:
        return explicit
    env = os.environ.get("FEPDASH_CONFIG")
    if env:
        return Path(env)
    candidate = repo_root() / "config.toml"
    return candidate if candidate.is_file() else None


def load_config(path: Optional[Path] = None) -> Config:
    """Load config, falling back to defaults for anything unspecified.

    A partial config.toml is legal and common: on GPU2 you typically only
    need ``[paths] runs_root`` and ``[gpu] allowed``.
    """
    cfg_path = _resolve_config_path(path)
    base = _defaults()
    if cfg_path is None or not cfg_path.is_file():
        return base

    with cfg_path.open("rb") as fh:
        raw = tomllib.load(fh)

    paths = raw.get("paths", {})
    gpu = raw.get("gpu", {})
    here = cfg_path.parent

    def _p(key: str, default: Path) -> Path:
        val = paths.get(key)
        if not val:
            return default
        # Relative entries resolve against the config file, not the cwd --
        # streamlit's cwd is wherever the operator launched it from.
        p = Path(val).expanduser()
        return p if p.is_absolute() else (here / p).resolve()

    runs_root = _p("runs_root", base.runs_root)
    db_path = _p("db_path", runs_root / "state.db")

    allowed = gpu.get("allowed", list(base.allowed_gpus))
    if isinstance(allowed, str):  # "0,1,2,3" is an easy thing to type
        allowed = [int(x) for x in allowed.replace(",", " ").split()]

    return Config(
        runs_root=runs_root,
        db_path=db_path,
        engines_dir=_p("engines_dir", base.engines_dir),
        inputs_root=_p("inputs_root", base.inputs_root),
        scripts_dir=_p("scripts_dir", base.scripts_dir),
        allowed_gpus=tuple(int(g) for g in allowed),
        enforce_gpu_locks=bool(gpu.get("enforce_locks", base.enforce_gpu_locks)),
    )
