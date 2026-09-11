"""fepdash.ui.common -- widgets shared across pages.

Kept deliberately small. Anything that reads or writes campaign state lives
in ``core``; this module only turns those values into Streamlit calls.
"""

from __future__ import annotations

import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fepdash.core import db as _db
from fepdash.core import gpu as _gpu
from fepdash.core import state as _state
from fepdash.core.config import Config, check_runs_root
from fepdash.core.models import Campaign, CampaignStatus, Method


_STATUS_ICON = {
    "queued": "⏳",
    "running": "🟢",
    "finished": "✅",
    "failed": "🔴",
    "killed": "⚫",
}

_LEG_ICON = {
    "pending": "·",
    "running": "🟢",
    "done": "✅",
    "failed": "🔴",
}


def page_header(title: str, subtitle: str = "") -> None:
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def require_usable_runs_root(cfg: Config) -> None:
    """Stop the page with a readable error if runs_root is unusable.

    Every page calls this before its first DB access. A config pointing at a
    directory you cannot create is the most likely first-run failure, and
    without this it surfaces as a PermissionError traceback from inside
    pathlib -- which never mentions the config file that actually caused it.
    """
    problem = check_runs_root(cfg)
    if problem is None:
        return

    st.error(problem)
    st.markdown(
        f"""
**Where this comes from**

| | |
|---|---|
| config file | `{os.environ.get("FEPDASH_CONFIG", "<none set — using defaults>")}` |
| `runs_root` | `{cfg.runs_root}` |

Point it somewhere you own, then reload:

```toml
[paths]
runs_root = "{Path.home() / "fep-runs"}"
```
"""
    )
    st.stop()


def fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "—"
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def fmt_timedelta(td: Optional[timedelta]) -> str:
    return fmt_duration(td.total_seconds()) if td is not None else "—"


# ---------------------------------------------------------------------------
# GPU board
# ---------------------------------------------------------------------------


def gpu_board(cfg: Config) -> None:
    """One row per allowed GPU: claimed by us, or busy according to the driver."""
    states = _gpu.gpu_inventory(cfg.runs_root, cfg.allowed_gpus)
    if not states:
        st.caption("No GPUs configured. Set `[gpu] allowed` in config.toml.")
        return

    for gpu_state in states:
        if gpu_state.is_claimed:
            icon = "🔒"
        elif gpu_state.looks_busy:
            icon = "⚠️"
        else:
            icon = "○"
        label = gpu_state.name or f"GPU {gpu_state.index}"
        st.markdown(
            f"{icon} **GPU {gpu_state.index}** — {gpu_state.describe()}  \n"
            f"<span style='color:#888;font-size:0.8em'>{label}</span>",
            unsafe_allow_html=True,
        )

    if any(s.looks_busy and not s.is_claimed for s in states):
        st.caption(
            "⚠️ = the driver reports work this dashboard did not start. "
            "GPU claims are advisory; they cannot see jobs launched outside."
        )


def gpu_selector(cfg: Config, *, key: str, default: Optional[list[int]] = None) -> list[int]:
    """Multiselect over allowed GPUs, annotated with who holds what."""
    states = {s.index: s for s in _gpu.gpu_inventory(cfg.runs_root, cfg.allowed_gpus)}

    def _label(idx: int) -> str:
        s = states.get(idx)
        if s is None:
            return f"GPU {idx}"
        if s.is_claimed:
            return f"GPU {idx} — 🔒 {s.claimed_by}"
        if s.looks_busy:
            return f"GPU {idx} — ⚠️ busy (not ours)"
        return f"GPU {idx} — free"

    free = [i for i, s in states.items() if not s.is_claimed and not s.looks_busy]
    return st.multiselect(
        "GPUs",
        options=list(cfg.allowed_gpus),
        default=default if default is not None else free,
        format_func=_label,
        key=key,
        help=(
            "OpenFE runs one leg per GPU. TMD schedules itself across whatever "
            "it is given via MPS — give it the whole set, not one card."
        ),
    )


# ---------------------------------------------------------------------------
# Campaign tables
# ---------------------------------------------------------------------------


def campaign_summary_table(rows: list[dict[str, Any]], engines: dict, cfg: Config) -> None:
    """Compact overview: status, engine, progress, age."""
    records = []
    for row in rows:
        campaign = _db.campaign_from_row(row, runs_root=cfg.runs_root)
        engine = engines.get(row["engine"])
        prog = _safe_progress(campaign, engine)
        records.append(
            {
                "": _STATUS_ICON.get(row["status"], "?"),
                "campaign": row["campaign_id"],
                "engine": row["engine"],
                "method": row["method"],
                "stage": prog.stage.value if prog and prog.stage else "—",
                "legs": f"{prog.done}/{prog.total}" if prog and prog.total else "—",
                "failed": prog.failed if prog else 0,
                "gpus": ",".join(str(g) for g in (row.get("gpus") or [])),
                "created": row.get("created_at", "")[:16].replace("T", " "),
            }
        )
    st.dataframe(
        pd.DataFrame(records),
        hide_index=True,
        width="stretch",
    )


def _safe_progress(campaign: Campaign, engine) -> Optional[_state.Progress]:
    """Progress for a campaign, tolerating a deleted directory or bad engine.

    The Runs table must render even when a campaign's directory was moved or
    an engine TOML was renamed -- otherwise one broken row hides every good
    one.
    """
    if engine is None or campaign.run_dir is None:
        return None
    try:
        jobs_glob = engine.stage(campaign.method, "plan").get("jobs_glob", "")
        return _state.progress(campaign, jobs_glob)
    except Exception:
        return None


def leg_table(progress: _state.Progress) -> pd.DataFrame:
    """Per-leg detail, sorted so the interesting rows are at the top."""
    order = {"failed": 0, "running": 1, "pending": 2, "done": 3}
    legs = sorted(progress.legs, key=lambda l: (order.get(l.status.value, 9), l.leg_id))
    return pd.DataFrame(
        [
            {
                "": _LEG_ICON.get(leg.status.value, "?"),
                "leg": leg.leg_id,
                "type": leg.leg_type or "—",
                # str, not int-or-str: a mixed-type column fails Arrow
                # serialisation and makes Streamlit log a conversion
                # traceback on every render.
                "gpu": str(leg.gpu) if leg.gpu is not None else "—",
                "duration": fmt_duration(leg.duration_s),
                "error": leg.error or "",
            }
            for leg in legs
        ]
    )


def progress_bar(progress: _state.Progress, n_gpus: int) -> None:
    cols = st.columns(5)
    cols[0].metric("done", progress.done)
    cols[1].metric("running", progress.running)
    cols[2].metric("pending", progress.pending)
    cols[3].metric("failed", progress.failed)
    cols[4].metric("ETA", fmt_timedelta(progress.eta(n_gpus)))

    if progress.total:
        st.progress(progress.fraction, text=f"{progress.done}/{progress.total} legs")

    mean = progress.mean_leg_seconds()
    if mean:
        st.caption(
            f"mean completed leg {fmt_duration(mean)} — ETA assumes legs are "
            f"interchangeable across {n_gpus} GPU(s), which understates complex "
            f"legs. Treat it as an order of magnitude."
        )
