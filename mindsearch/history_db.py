"""DuckDB helper for MindSearch research history."""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = "/app/data/mindsearch.duckdb"

_DDL = """
CREATE TABLE IF NOT EXISTS researches (
    id       TEXT PRIMARY KEY,
    title    TEXT NOT NULL,
    saved_at TIMESTAMP NOT NULL,
    data     TEXT NOT NULL
);
"""


def _connect():
    import duckdb
    conn = duckdb.connect(DB_PATH)
    conn.execute(_DDL)
    return conn


def list_researches():
    """Return list of {id, title, saved_at} dicts, newest first."""
    conn = _connect()
    rows = conn.execute(
        "SELECT id, title, saved_at FROM researches ORDER BY saved_at DESC"
    ).fetchall()
    conn.close()
    return [
        {
            "id": r[0],
            "title": r[1],
            "saved_at": r[2].isoformat() if hasattr(r[2], "isoformat") else str(r[2]),
        }
        for r in rows
    ]


def save_research(title: str, data) -> str:
    """Save a research session. Returns the new id."""
    rid = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    conn = _connect()
    conn.execute(
        "INSERT INTO researches (id, title, saved_at, data) VALUES (?, ?, ?, ?)",
        [rid, title, now, json.dumps(data)],
    )
    conn.close()
    return rid


def load_research(rid: str) -> dict | None:
    """Return full research record or None if not found."""
    conn = _connect()
    row = conn.execute(
        "SELECT id, title, saved_at, data FROM researches WHERE id = ?", [rid]
    ).fetchone()
    conn.close()
    if not row:
        return None
    return {
        "id": row[0],
        "title": row[1],
        "saved_at": row[2].isoformat() if hasattr(row[2], "isoformat") else str(row[2]),
        "data": json.loads(row[3]),
    }


def delete_research(rid: str) -> bool:
    """Delete a research by id. Returns True if a row was deleted."""
    conn = _connect()
    conn.execute("DELETE FROM researches WHERE id = ?", [rid])
    # DuckDB doesn't expose rowcount easily; check existence
    still = conn.execute("SELECT COUNT(*) FROM researches WHERE id = ?", [rid]).fetchone()[0]
    conn.close()
    return still == 0
