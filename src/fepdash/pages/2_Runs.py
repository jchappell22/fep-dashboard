"""Runs page -- live progress for one campaign, and the controls to stop it.

Everything shown here is derived from the campaign directory on each render,
so the page is correct even if the dashboard was restarted, is running in
two browser tabs, or was never running while the campaign progressed.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fepdash.core import db as _db  # noqa: E402
from fepdash.core import launcher  # noqa: E402
from fepdash.core import polling  # noqa: E402
from fepdash.core import state as _state  # noqa: E402
from fepdash.core.config import load_config  # noqa: E402
from fepdash.core.engines.base import load_engines  # noqa: E402
from fepdash.core.models import CampaignStatus  # noqa: E402
from fepdash.ui.common import (  # noqa: E402
    require_usable_runs_root,
    fmt_duration,
    leg_table,
    page_header,
    progress_bar,
)

st.set_page_config(page_title="Runs — FEP dashboard", page_icon="📊", layout="wide")


@st.cache_data(ttl=60, show_spinner=False)
def _read_iteration(path_str: str, _mtime: float):
    """Sampler iteration from a live NetCDF, cached on (path, mtime).

    Defined up here because the page body is executed top-to-bottom on every
    rerun -- a definition below its call site is a NameError, not a hoisted
    function.

    Caching on mtime means an in-flight simulation re-reads (its file keeps
    changing) while a finished leg is read once. Without the cache this
    would open a NetCDF that a running simulation is writing, on every
    single widget interaction.
    """
    return _state.openmm_iteration(Path(path_str))


cfg = load_config()
require_usable_runs_root(cfg)
_db.init_db(cfg.db_path)
polling.poll_active(cfg)

page_header("Runs", "Live campaign progress, straight off the filesystem.")

rows = _db.list_campaigns(cfg.db_path, limit=200)
if not rows:
    st.info(
        "No campaigns yet. Generate synthetic ones to exercise this page without "
        f"a GPU:\n\n```\npython -m fepdash.core.fixtures --runs-root {cfg.runs_root}\n```"
    )
    st.stop()

engines = load_engines(cfg.engines_dir)

labels = {
    r["campaign_id"]: f"{r['campaign_id']}  ({r['status']}, {r['engine']} {r['method']})"
    for r in rows
}
selected_id = st.selectbox(
    "Campaign", options=list(labels), format_func=lambda c: labels[c]
)
row = _db.get_campaign(selected_id, cfg.db_path)
campaign = _db.campaign_from_row(row, runs_root=cfg.runs_root)
engine = engines.get(row["engine"])

status = CampaignStatus(row["status"])

# ---------------------------------------------------------------------------
# Header + controls
# ---------------------------------------------------------------------------

head = st.columns([3, 1, 1, 1])
head[0].markdown(
    f"### {campaign.name}\n"
    f"`{campaign.run_dir}`  \n"
    f"engine **{row['engine']}** · method **{row['method']}** · "
    f"GPUs **{', '.join(str(g) for g in (row.get('gpus') or [])) or '—'}** · "
    f"pid **{row.get('pid') or '—'}**"
)
head[1].metric("status", status.value)

if not status.is_terminal:
    if head[2].button("Kill", type="primary", help="SIGTERM the driver's whole process group, then SIGKILL."):
        result = launcher.kill_campaign(selected_id, cfg)
        st.warning(f"kill: {result}")
        st.rerun()
else:
    if head[2].button("Resume", help="Re-run driver.sh. Completed legs are skipped."):
        try:
            msg = launcher.resume_campaign(selected_id, cfg)
        except launcher.LaunchError as exc:
            st.error(str(exc))
        else:
            st.success(msg)
            st.rerun()

auto = head[3].toggle("Auto-refresh", value=False, help="Re-read the campaign every 20 s.")

if row.get("notes"):
    st.info(row["notes"])

if engine is None:
    st.error(
        f"Engine '{row['engine']}' is no longer defined in {cfg.engines_dir}, so "
        f"leg progress cannot be interpreted. The campaign directory is intact."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------

jobs_glob = engine.stage(campaign.method, "plan").get("jobs_glob", "")
progress = _state.progress(campaign, jobs_glob)

stage = progress.stage
if stage is not None:
    st.markdown(f"**stage:** `{stage.value}`")
    if stage.value == "plan":
        st.caption(
            "Planning generates partial charges for every ligand before any GPU "
            "work starts. On a large series this is a long CPU-bound step — a "
            "campaign sitting here is not stuck."
        )

progress_bar(progress, n_gpus=max(len(row.get("gpus") or []), 1))

if progress.total == 0:
    st.warning(
        "No legs yet. Either planning has not finished, or it produced no job "
        f"files matching `plan/{jobs_glob}`. Check the driver log below."
    )

tab_legs, tab_log, tab_inspect = st.tabs(["Legs", "Driver log", "Inspect a leg"])

with tab_legs:
    if progress.legs:
        st.dataframe(leg_table(progress), hide_index=True, width="stretch")
    failed = [l for l in progress.legs if l.status.value == "failed"]
    if failed:
        st.caption(
            f"{len(failed)} failed leg(s). Re-running the campaign (**Resume**) "
            f"retries them — completed legs are skipped."
        )

with tab_log:
    log_choice = st.radio(
        "stream", ["stdout", "stderr"], horizontal=True, label_visibility="collapsed"
    )
    log_path = campaign.run_dir / f"{log_choice}.log"
    text = _state.tail(log_path, n=400)
    st.code(text or f"({log_path.name} is empty)", language="text")
    st.caption(f"`{log_path}` — last 400 lines")

with tab_inspect:
    running = [l for l in progress.legs if l.status.value in ("running", "failed")]
    if not running:
        st.caption("No running or failed legs to inspect.")
    else:
        leg_id = st.selectbox("Leg", [l.leg_id for l in running])
        leg = next(l for l in progress.legs if l.leg_id == leg_id)

        cols = st.columns(3)
        cols[0].metric("status", leg.status.value)
        cols[1].metric("gpu", leg.gpu if leg.gpu is not None else "—")
        cols[2].metric("duration", fmt_duration(leg.duration_s))

        ncs = _state.find_simulation_ncs(campaign, leg_id)
        if ncs:
            st.markdown(f"**{len(ncs)}** simulation file(s) under `work/{leg_id}/`")
            newest = ncs[0]
            age = _state.age_seconds(newest)
            if age is not None:
                st.caption(f"last write to `{newest.name}`: {fmt_duration(age)} ago")
                if age > 3600 and leg.status.value == "running":
                    st.warning(
                        "No write for over an hour. The leg may be stalled rather "
                        "than slow — check the leg log and nvidia-smi."
                    )
            # Reading the sampler iteration opens a NetCDF a live simulation is
            # writing, so it is behind a button and cached, never on every rerun.
            if st.button("Read sampler iteration (slow)"):
                iteration = _read_iteration(str(newest), newest.stat().st_mtime)
                st.metric("iteration", iteration if iteration is not None else "unreadable")

        if leg.log_path:
            st.code(_state.tail(leg.log_path, n=300), language="text")
            st.caption(f"`{leg.log_path}` — last 300 lines")


if auto and not status.is_terminal:
    time.sleep(20)
    st.rerun()
