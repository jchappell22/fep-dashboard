"""fep-dashboard -- home page.

Run with::

    streamlit run src/fepdash/app.py

The home page is a status board, not a control panel: what is running, what
the cards are doing, and whether the engine definitions loaded. Everything
that changes state lives on a page you have to navigate to on purpose.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make `fepdash` importable when Streamlit runs this file by path rather
# than as an installed package -- the common case on a box where the repo
# was cloned and not pip-installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fepdash.core import db as _db  # noqa: E402
from fepdash.core import gpu as _gpu  # noqa: E402
from fepdash.core import polling  # noqa: E402
from fepdash.core.config import load_config  # noqa: E402
from fepdash.core.engines.base import load_engines  # noqa: E402
from fepdash.ui.common import (  # noqa: E402
    campaign_summary_table,
    gpu_board,
    page_header,
    require_usable_runs_root,
)

st.set_page_config(page_title="FEP dashboard", page_icon="🧬", layout="wide")


def main() -> None:
    cfg = load_config()
    page_header(
        "FEP dashboard",
        "Free energy campaigns across OpenFE and TMD, on this box's GPUs.",
    )
    # Before any DB access: a runs_root we cannot write to must produce a
    # readable message, not a PermissionError traceback out of pathlib.
    require_usable_runs_root(cfg)
    _db.init_db(cfg.db_path)

    # Notice any driver that exited since the last page load. Cheap: one
    # kill(pid, 0) per active campaign.
    transitions = polling.poll_active(cfg)

    if transitions:
        for campaign_id, status in transitions.items():
            st.toast(f"{campaign_id} -> {status}")

    engines = load_engines(cfg.engines_dir)
    broken = [e for e in engines.values() if not e.methods]
    if broken:
        for engine in broken:
            st.error(f"engine definition failed to load: {engine.label}")
    if not engines:
        st.error(
            f"No engine definitions found in {cfg.engines_dir}. "
            "The dashboard cannot launch anything without them."
        )

    left, right = st.columns([2, 1])

    with left:
        st.subheader("Campaigns")
        rows = _db.list_campaigns(cfg.db_path, limit=50)
        if not rows:
            st.info(
                "No campaigns yet. Use **Launch** to start one, or generate a "
                "synthetic campaign to try the UI without burning GPU time:\n\n"
                f"```\npython -m fepdash.core.fixtures --runs-root {cfg.runs_root}\n```"
            )
        else:
            campaign_summary_table(rows, engines, cfg)

    with right:
        st.subheader("GPUs")
        gpu_board(cfg)

        st.subheader("Engines")
        for engine in engines.values():
            if not engine.methods:
                continue
            st.markdown(
                f"**{engine.name}** — {', '.join(engine.methods)}  \n"
                f"<span style='color:#888;font-size:0.85em'>env "
                f"<code>{engine.conda_env}</code> · {engine.source.name}</span>",
                unsafe_allow_html=True,
            )

    with st.expander("Where things live"):
        st.markdown(
            f"""
| | |
|---|---|
| runs root | `{cfg.runs_root}` |
| state DB | `{cfg.db_path}` |
| engine defs | `{cfg.engines_dir}` |
| inputs | `{cfg.inputs_root}` |
| allowed GPUs | `{list(cfg.allowed_gpus)}` |
| GPU locks enforced | `{cfg.enforce_gpu_locks}` |

Set `FEPDASH_CONFIG=/path/to/config.toml` to change any of these.
"""
        )


main()
