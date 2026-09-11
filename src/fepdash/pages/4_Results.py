"""Results page -- the ranked ligands, the edge table, and how much to trust it.

Both engines are normalised into the same two tables (see core/results.py),
so nothing on this page branches on the engine.

The cycle-closure panel is the point of the page. A ranking by itself always
looks authoritative; the residuals tell you which edges the network disagrees
with itself about, which is where a bad atom mapping or an unconverged leg
shows up.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fepdash.core import db as _db  # noqa: E402
from fepdash.core import results as _results  # noqa: E402
from fepdash.core import state as _state  # noqa: E402
from fepdash.core.config import load_config  # noqa: E402
from fepdash.core.engines.base import load_engines  # noqa: E402
from fepdash.ui.common import page_header, require_usable_runs_root  # noqa: E402

st.set_page_config(page_title="Results — FEP dashboard", page_icon="📈", layout="wide")

cfg = load_config()
require_usable_runs_root(cfg)
_db.init_db(cfg.db_path)

page_header("Results", "Per-ligand ΔG, per-edge ΔΔG, and network consistency.")

rows = _db.list_campaigns(cfg.db_path, limit=200)
if not rows:
    st.info("No campaigns yet.")
    st.stop()

engines = load_engines(cfg.engines_dir)

labels = {r["campaign_id"]: f"{r['campaign_id']}  ({r['status']})" for r in rows}
selected_id = st.selectbox("Campaign", list(labels), format_func=lambda c: labels[c])
row = _db.get_campaign(selected_id, cfg.db_path)
campaign = _db.campaign_from_row(row, runs_root=cfg.runs_root)
engine = engines.get(row["engine"])

if engine is None:
    st.error(f"Engine '{row['engine']}' is not defined; cannot locate its tables.")
    st.stop()

tables = _results.load_all(campaign, engine, campaign.method)

if not tables:
    paths = _results.table_paths(campaign, engine, campaign.method)
    st.warning("No result tables on disk yet. Expected:")
    for kind, (path, fmt) in paths.items():
        st.markdown(f"- `{path}` ({fmt})")
    gather_spec = engine.stage(campaign.method, "gather")
    if gather_spec.cmds:
        st.caption(
            "These are written by the driver's gather stage, which runs after "
            "the legs. You can also produce them early by running the gather "
            "commands by hand — they use `--allow-partial`."
        )
    else:
        st.caption(f"`{engine.name}` writes these itself while the run proceeds.")
    st.stop()


# ---------------------------------------------------------------------------
# Per-ligand ranking
# ---------------------------------------------------------------------------

dg = tables.get("dg")
if dg is not None:
    st.subheader("Ligands, ranked")
    for problem in dg.problems:
        st.warning(problem)
    if dg.ok:
        frame = dg.normalised
        st.dataframe(
            frame.style.format({"dG": "{:.2f}", "uncertainty": "{:.2f}"}),
            hide_index=True,
            width="stretch",
        )
        # A chart with error bars, since a ranking without them invites
        # over-reading gaps that are inside the noise.
        chart = frame.dropna(subset=["dG"]).set_index("ligand")[["dG"]]
        st.bar_chart(chart, horizontal=True, height=max(240, 28 * len(chart)))
        if frame["uncertainty"].notna().any():
            median_err = float(frame["uncertainty"].median())
            st.caption(
                f"Median uncertainty {median_err:.2f} kcal/mol — treat any two "
                f"ligands closer than about {2 * median_err:.1f} kcal/mol as "
                f"unranked relative to each other."
            )
    else:
        st.dataframe(dg.raw, hide_index=True, width="stretch")
    st.caption(f"source: `{dg.path}`")


# ---------------------------------------------------------------------------
# Per-edge table + consistency
# ---------------------------------------------------------------------------

ddg = tables.get("ddg")
if ddg is not None:
    st.subheader("Edges")
    for problem in ddg.problems:
        st.warning(problem)

    if ddg.ok:
        st.dataframe(
            ddg.normalised.style.format({"ddG": "{:.2f}", "uncertainty": "{:.2f}"}),
            hide_index=True,
            width="stretch",
        )

        closure = _results.cycle_closure(ddg.normalised)
        st.subheader("Network consistency")
        if closure is None:
            st.caption(
                "The edge network has no cycles (or too few edges), so there is "
                "nothing to close. A radial or spanning-tree network cannot be "
                "checked this way — add redundant edges if you want this signal."
            )
        else:
            worst = closure.head(10)[
                ["ligand_i", "ligand_j", "ddG", "ddG_fitted", "residual"]
            ]
            st.dataframe(
                worst.style.format(
                    {"ddG": "{:.2f}", "ddG_fitted": "{:.2f}", "residual": "{:+.2f}"}
                ),
                hide_index=True,
                width="stretch",
            )
            largest = float(closure["residual"].abs().max())
            rms = float((closure["residual"] ** 2).mean() ** 0.5)
            cols = st.columns(2)
            cols[0].metric("RMS residual", f"{rms:.2f} kcal/mol")
            cols[1].metric("worst edge", f"{largest:.2f} kcal/mol")
            if largest > 1.5:
                st.warning(
                    f"This network miscloses by up to {largest:.2f} kcal/mol. "
                    f"That usually means a poor atom mapping or an unconverged "
                    f"leg somewhere in the affected loop, not a real affinity "
                    f"difference — investigate before trusting the ranking above."
                )
            st.caption(
                "Residual = measured ΔΔG minus the value implied by a "
                "least-squares fit of per-ligand free energies to the whole "
                "network.\n\n"
                "**Read these as flagging a loop, not an edge.** Least squares "
                "cannot tell which member of an inconsistent cycle is wrong, so "
                "it spreads the discrepancy across all of them — in a bare "
                "triangle every residual comes out equal. Blame only concentrates "
                "on a single edge when that edge sits in several overlapping "
                "cycles, so the top row here is a starting point, not a verdict."
            )
    else:
        st.dataframe(ddg.raw, hide_index=True, width="stretch")
    st.caption(f"source: `{ddg.path}`")


# ---------------------------------------------------------------------------
# Raw + download
# ---------------------------------------------------------------------------

raw = tables.get("raw")
if raw is not None:
    with st.expander("Raw per-repeat results"):
        st.dataframe(raw.raw, hide_index=True, width="stretch")
        st.caption(f"source: `{raw.path}`")

st.subheader("Download")
cols = st.columns(len(tables) or 1)
for i, (kind, table) in enumerate(tables.items()):
    frame = table.normalised if table.normalised is not None else table.raw
    cols[i].download_button(
        f"{kind}.csv",
        data=frame.to_csv(index=False).encode(),
        file_name=f"{selected_id}_{kind}.csv",
        mime="text/csv",
    )
