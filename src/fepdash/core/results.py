"""fepdash.core.results -- read both engines' result tables into one shape.

The two engines disagree about nearly everything cosmetic:

==============  =========================  =========================
                OpenFE                     TMD
==============  =========================  =========================
format          tab-separated              comma-separated
per-edge file   ``ddg.tsv`` (gathered/)    ``ddg_results.csv`` (results/)
per-ligand      ``dg.tsv``                 ``dg_results.csv``
produced by     a separate gather step     written during the run
==============  =========================  =========================

...and agree about what matters: an edge table of (ligand_i, ligand_j, ddG,
uncertainty) and a ligand table of (ligand, dG, uncertainty).

This module normalises both into those two shapes so the Results page has
no engine-specific branches. Column *names* are matched case-insensitively
against a list of known aliases rather than by position, because position
is exactly the sort of thing that changes between versions -- and Jacob has
already been bitten by a column-merge bug in a different project.

If a table cannot be recognised, the raw DataFrame is returned alongside an
explanation rather than a guess. A silently mis-mapped dG column produces a
ranking that looks plausible and is wrong, which is the worst outcome here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd


# Column aliases, lowercased. Order within a tuple is preference order.
_LIGAND_I = ("ligand_i", "ligand_a", "mol_i", "mol_a", "lig_i", "from", "source")
_LIGAND_J = ("ligand_j", "ligand_b", "mol_j", "mol_b", "lig_j", "to", "target")
_LIGAND = ("ligand", "ligand_name", "mol", "molecule", "name", "smiles_name")
_DDG = ("ddg(i->j) (kcal/mol)", "ddg (kcal/mol)", "ddg", "ddg_kcal", "pred_ddg", "dddg")
_DG = ("dg(mle) (kcal/mol)", "dg (kcal/mol)", "dg", "dg_kcal", "pred_dg", "free_energy")
_UNC = (
    "uncertainty (kcal/mol)",
    "uncertainty",
    "ddg_error",
    "dg_error",
    "err",
    "error",
    "std",
    "stderr",
    "sem",
)


@dataclass
class ResultTable:
    """A parsed table plus what we could and could not make of it."""

    kind: str  # "ddg" | "dg"
    path: Path
    raw: pd.DataFrame
    normalised: Optional[pd.DataFrame] = None
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.normalised is not None and not self.normalised.empty


def _find_column(df: pd.DataFrame, aliases: tuple[str, ...]) -> Optional[str]:
    """Match a column by alias, case- and whitespace-insensitively."""
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for alias in aliases:
        if alias in lookup:
            return lookup[alias]
    # Substring fallback, longest match first, so "DDG (kcal/mol)" is found
    # even if the exact spelling drifts. Only used when no alias hit.
    for alias in aliases:
        hits = [orig for low, orig in lookup.items() if alias in low]
        if len(hits) == 1:
            return hits[0]
    return None


def _read_any(path: Path, fmt: str) -> pd.DataFrame:
    sep = "," if fmt == "csv" else "\t"
    # `sep=None` + python engine sniffs the delimiter, which rescues the
    # case where an engine's output is actually the other format. The
    # declared format is tried first because sniffing can be wrong on a
    # one-column file.
    try:
        df = pd.read_csv(path, sep=sep)
        if df.shape[1] > 1:
            return df
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        pass
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def load_ddg(path: Path, fmt: str = "tsv") -> ResultTable:
    """Per-edge relative free energies."""
    raw = _read_any(path, fmt)
    table = ResultTable(kind="ddg", path=path, raw=raw)
    if raw.empty:
        table.problems.append(f"{path.name} is empty or unreadable")
        return table

    col_i = _find_column(raw, _LIGAND_I)
    col_j = _find_column(raw, _LIGAND_J)
    col_ddg = _find_column(raw, _DDG)
    col_unc = _find_column(raw, _UNC)

    missing = [
        label
        for label, col in (("ligand_i", col_i), ("ligand_j", col_j), ("ddG", col_ddg))
        if col is None
    ]
    if missing:
        table.problems.append(
            f"could not find {', '.join(missing)} among columns "
            f"{list(raw.columns)} -- showing the raw table instead. Add the "
            f"real column name to _LIGAND_I/_LIGAND_J/_DDG in core/results.py."
        )
        return table

    out = pd.DataFrame(
        {
            "ligand_i": raw[col_i].astype(str),
            "ligand_j": raw[col_j].astype(str),
            "ddG": pd.to_numeric(raw[col_ddg], errors="coerce"),
        }
    )
    out["uncertainty"] = (
        pd.to_numeric(raw[col_unc], errors="coerce") if col_unc else pd.NA
    )
    if col_unc is None:
        table.problems.append("no uncertainty column found; error bars unavailable")

    dropped = int(out["ddG"].isna().sum())
    if dropped:
        table.problems.append(f"{dropped} edge(s) had a non-numeric ddG and were kept as NaN")

    table.normalised = out
    return table


def load_dg(path: Path, fmt: str = "tsv") -> ResultTable:
    """Per-ligand absolute (or MLE-centred) free energies."""
    raw = _read_any(path, fmt)
    table = ResultTable(kind="dg", path=path, raw=raw)
    if raw.empty:
        table.problems.append(f"{path.name} is empty or unreadable")
        return table

    col_lig = _find_column(raw, _LIGAND)
    col_dg = _find_column(raw, _DG)
    col_unc = _find_column(raw, _UNC)

    missing = [
        label for label, col in (("ligand", col_lig), ("dG", col_dg)) if col is None
    ]
    if missing:
        table.problems.append(
            f"could not find {', '.join(missing)} among columns "
            f"{list(raw.columns)} -- showing the raw table instead."
        )
        return table

    out = pd.DataFrame(
        {
            "ligand": raw[col_lig].astype(str),
            "dG": pd.to_numeric(raw[col_dg], errors="coerce"),
        }
    )
    out["uncertainty"] = (
        pd.to_numeric(raw[col_unc], errors="coerce") if col_unc else pd.NA
    )

    # Rank ascending: more negative dG binds more tightly.
    out = out.sort_values("dG", na_position="last").reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))

    if out["dG"].notna().any():
        # OpenFE's MLE dG values are centred on zero and only meaningful
        # relative to each other. Offering the spread makes that obvious.
        table.problems.append(
            "note: dG values from an MLE network are centred near zero -- "
            "compare ligands to each other, not to an absolute scale"
            if "mle" in str(col_dg).lower()
            else ""
        )
        table.problems = [p for p in table.problems if p]

    table.normalised = out
    return table


# ---------------------------------------------------------------------------
# Locating the tables
# ---------------------------------------------------------------------------


def table_paths(campaign, engine, method) -> dict[str, tuple[Path, str]]:
    """Where this engine's tables live, as ``{kind: (path, format)}``.

    Handles the structural difference that OpenFE writes into a gather
    directory the dashboard owns, while TMD writes into its own output
    directory as it runs.
    """
    spec = engine.stage(method, "gather")
    fmt = spec.get("table_format", "tsv")
    base = (
        campaign.results_dir
        if spec.get("tables_in") == "results_dir"
        else campaign.gathered_dir
    )
    out: dict[str, tuple[Path, str]] = {}
    for kind, key in (("ddg", "ddg_table"), ("dg", "dg_table"), ("raw", "raw_table")):
        filename = spec.get(key, "")
        if filename:
            out[kind] = (base / filename, fmt)
    return out


def load_all(campaign, engine, method) -> dict[str, ResultTable]:
    """Load whatever tables exist. Absent files are simply omitted."""
    loaded: dict[str, ResultTable] = {}
    for kind, (path, fmt) in table_paths(campaign, engine, method).items():
        if not path.is_file():
            continue
        if kind == "ddg":
            loaded[kind] = load_ddg(path, fmt)
        elif kind == "dg":
            loaded[kind] = load_dg(path, fmt)
        else:
            loaded[kind] = ResultTable(kind="raw", path=path, raw=_read_any(path, fmt))
    return loaded


def cycle_closure(ddg: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Per-edge deviation from a consistent set of node free energies.

    A cheap, engine-independent sanity check: least-squares fit node values
    to the measured edges, then report each edge's residual. Large residuals
    flag ddG values inconsistent with the rest of the network -- usually a
    bad atom mapping or an unconverged leg -- which is exactly what you want
    to spot before trusting a ranking.

    IMPORTANT LIMITATION -- it localises to a *cycle*, not to an edge.
    Least squares has no way to know which member of an inconsistent cycle
    is the liar, so it spreads the discrepancy over all of them. In a bare
    triangle (a->b, b->c, a->c) with one bad edge, all three residuals come
    back equal in magnitude. Only where an edge sits in several overlapping
    cycles does the fit start to concentrate blame on it. So read a large
    residual as "something in this loop is wrong", and use the redundancy of
    your network to narrow it down -- do not assume the worst-listed edge is
    the culprit.

    Returns None when the edge table is too small to be over-determined
    (a spanning tree has no cycles, so there is nothing to close).
    """
    try:
        import numpy as np
    except ImportError:
        return None

    edges = ddg.dropna(subset=["ddG"])
    if len(edges) < 2:
        return None

    ligands = sorted(set(edges["ligand_i"]) | set(edges["ligand_j"]))
    if len(edges) <= len(ligands) - 1:
        return None  # a tree: no cycles, nothing to close

    index = {name: k for k, name in enumerate(ligands)}
    A = np.zeros((len(edges) + 1, len(ligands)))
    b = np.zeros(len(edges) + 1)
    for row, (_, edge) in enumerate(edges.iterrows()):
        A[row, index[edge["ligand_j"]]] = 1.0
        A[row, index[edge["ligand_i"]]] = -1.0
        b[row] = float(edge["ddG"])
    # Gauge fix: the network only determines differences, so pin the mean.
    A[-1, :] = 1.0
    b[-1] = 0.0

    try:
        node_dg, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None

    predicted = [
        node_dg[index[e["ligand_j"]]] - node_dg[index[e["ligand_i"]]]
        for _, e in edges.iterrows()
    ]
    out = edges.copy()
    out["ddG_fitted"] = predicted
    out["residual"] = out["ddG"] - out["ddG_fitted"]
    return out.sort_values("residual", key=lambda s: s.abs(), ascending=False)
