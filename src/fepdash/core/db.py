"""fepdash.core.db -- the campaigns table.

Scope, deliberately narrow
--------------------------

The DB holds **only what the filesystem cannot tell us**: the operator's
intent, the pid, and the exit code. Leg progress, results, and stage are all
derived by scanning the campaign directory on every read (see ``state.py``).

That split is what makes the dashboard safe to restart, and what stops the
DB from ever disagreeing with the disk about whether a leg finished.

Concurrency: WAL mode, and every writer opens/commits/closes. Streamlit runs
each page in its own thread and reruns the script on every widget touch, so
a long-lived connection shared across reruns is a reliable way to get
``SQLite objects created in a thread can only be used in that same thread``.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Optional

from .models import Campaign, CampaignStatus, now_iso


SCHEMA = """
CREATE TABLE IF NOT EXISTS campaigns (
    campaign_id   TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    engine        TEXT NOT NULL,
    method        TEXT NOT NULL,
    status        TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    started_at    TEXT,
    finished_at   TEXT,
    pid           INTEGER,
    exit_code     INTEGER,
    run_dir       TEXT NOT NULL,
    gpus          TEXT NOT NULL DEFAULT '[]',
    manifest_json TEXT NOT NULL DEFAULT '{}',
    notes         TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_campaigns_status ON campaigns(status);
CREATE INDEX IF NOT EXISTS idx_campaigns_created ON campaigns(created_at DESC);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    # WAL lets the Runs page read while the launcher writes.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: Path) -> None:
    conn = connect(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------


def insert_campaign(campaign: Campaign, db_path: Path) -> None:
    """Insert as QUEUED. Called *before* Popen so a spawn failure is visible."""
    conn = connect(db_path)
    try:
        conn.execute(
            "INSERT INTO campaigns (campaign_id, name, engine, method, status,"
            " created_at, run_dir, gpus, manifest_json, notes)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                campaign.campaign_id,
                campaign.name,
                campaign.engine,
                campaign.method.value,
                CampaignStatus.QUEUED.value,
                campaign.created_at,
                str(campaign.run_dir),
                json.dumps(campaign.gpus),
                json.dumps(campaign.to_dict()),
                campaign.notes,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def mark_running(campaign_id: str, pid: int, db_path: Path) -> None:
    _update(
        db_path,
        "UPDATE campaigns SET status = ?, pid = ?, started_at = ? WHERE campaign_id = ?",
        (CampaignStatus.RUNNING.value, pid, now_iso(), campaign_id),
    )


def mark_terminal(
    campaign_id: str,
    status: CampaignStatus,
    db_path: Path,
    *,
    exit_code: Optional[int] = None,
    notes: Optional[str] = None,
) -> None:
    if notes is None:
        _update(
            db_path,
            "UPDATE campaigns SET status = ?, finished_at = ?, exit_code = ?"
            " WHERE campaign_id = ?",
            (status.value, now_iso(), exit_code, campaign_id),
        )
    else:
        _update(
            db_path,
            "UPDATE campaigns SET status = ?, finished_at = ?, exit_code = ?,"
            " notes = ? WHERE campaign_id = ?",
            (status.value, now_iso(), exit_code, notes, campaign_id),
        )


def _update(db_path: Path, sql: str, args: tuple) -> None:
    conn = connect(db_path)
    try:
        conn.execute(sql, args)
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def list_campaigns(db_path: Path, *, limit: int = 200) -> list[dict[str, Any]]:
    conn = connect(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM campaigns ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    finally:
        conn.close()
    return [_row_to_dict(r) for r in rows]


def get_campaign(campaign_id: str, db_path: Path) -> Optional[dict[str, Any]]:
    conn = connect(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM campaigns WHERE campaign_id = ?", (campaign_id,)
        ).fetchone()
    finally:
        conn.close()
    return _row_to_dict(row) if row else None


def active_campaigns(db_path: Path) -> list[dict[str, Any]]:
    """Rows the poller needs to look at: queued or running."""
    conn = connect(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM campaigns WHERE status IN (?, ?)",
            (CampaignStatus.QUEUED.value, CampaignStatus.RUNNING.value),
        ).fetchall()
    finally:
        conn.close()
    return [_row_to_dict(r) for r in rows]


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    for key in ("gpus", "manifest_json"):
        raw = d.get(key)
        if isinstance(raw, str):
            try:
                d[key] = json.loads(raw)
            except json.JSONDecodeError:
                d[key] = [] if key == "gpus" else {}
    return d


def campaign_from_row(
    row: dict[str, Any], *, runs_root: Optional[Path] = None
) -> Campaign:
    """Rebuild the domain object from a row's manifest snapshot.

    ``runs_root`` enables recovery from a moved campaign tree. The recorded
    ``run_dir`` is absolute, so relocating ``runs_root`` (a new disk, a
    restored backup, a different mount on a rebuilt box) would otherwise
    leave every row pointing at a path that no longer exists -- the campaign
    would render with no legs and no results while the directory sat intact
    somewhere else. When the recorded path is gone but
    ``<runs_root>/<campaign_id>`` exists, use that instead.
    """
    manifest = row.get("manifest_json") or {}
    if manifest:
        campaign = Campaign.from_dict(manifest)
        campaign.run_dir = _resolve_run_dir(
            campaign.run_dir, row["campaign_id"], runs_root
        )
        return campaign
    # Fall back to the columns if the manifest is missing or corrupt, so a
    # damaged row still renders in the Runs table instead of raising.
    return Campaign(
        campaign_id=row["campaign_id"],
        name=row.get("name", row["campaign_id"]),
        engine=row.get("engine", "?"),
        method=row.get("method", "rbfe"),
        protein=Path("."),
        ligands=Path("."),
        gpus=row.get("gpus") or [],
        run_dir=_resolve_run_dir(Path(row["run_dir"]), row["campaign_id"], runs_root),
        created_at=row.get("created_at", ""),
    )


def _resolve_run_dir(
    recorded: Optional[Path], campaign_id: str, runs_root: Optional[Path]
) -> Optional[Path]:
    """Prefer the recorded path; fall back to ``<runs_root>/<campaign_id>``."""
    if recorded is not None and Path(recorded).is_dir():
        return Path(recorded)
    if runs_root is not None:
        candidate = runs_root / campaign_id
        if candidate.is_dir():
            return candidate
    return recorded
