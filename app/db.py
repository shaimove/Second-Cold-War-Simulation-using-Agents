"""SQLite storage for saved scenario runs and the LLM response cache.

Two tables:
- scenario_runs: persistent record of every simulation
- llm_cache:    deterministic-key cache for repeated agent calls
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from . import config as _config_mod
from .utils import utcnow_iso


SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS scenario_runs (
        run_id          TEXT PRIMARY KEY,
        created_at      TEXT NOT NULL,
        seed            TEXT NOT NULL,
        scenario_mode   TEXT NOT NULL,
        scenario_title  TEXT NOT NULL,
        full_json       TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS llm_cache (
        cache_key       TEXT PRIMARY KEY,
        created_at      TEXT NOT NULL,
        model           TEXT NOT NULL,
        agent_name      TEXT NOT NULL,
        response_json   TEXT NOT NULL
    )
    """,
]


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


@contextmanager
def connect(path: Optional[str] = None) -> Iterator[sqlite3.Connection]:
    db_path = path or _config_mod.CONFIG.sqlite_path
    _ensure_parent_dir(db_path)
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(path: Optional[str] = None) -> None:
    with connect(path) as conn:
        cur = conn.cursor()
        for stmt in SCHEMA:
            cur.execute(stmt)


# --- Scenario runs ---------------------------------------------------------


def save_scenario_run(
    run_id: str,
    seed: str,
    scenario_mode: str,
    scenario_title: str,
    full_json: Dict[str, Any],
    path: Optional[str] = None,
) -> None:
    init_db(path)
    with connect(path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO scenario_runs
                (run_id, created_at, seed, scenario_mode,
                 scenario_title, full_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                utcnow_iso(),
                seed,
                scenario_mode,
                scenario_title,
                json.dumps(full_json),
            ),
        )


def list_scenario_runs(
    limit: int = 50, path: Optional[str] = None
) -> List[Dict[str, Any]]:
    init_db(path)
    with connect(path) as conn:
        rows = conn.execute(
            """
            SELECT run_id, created_at, seed, scenario_mode, scenario_title
            FROM scenario_runs
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def load_scenario_run(
    run_id: str, path: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    init_db(path)
    with connect(path) as conn:
        row = conn.execute(
            "SELECT full_json FROM scenario_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        try:
            return json.loads(row["full_json"])
        except Exception:
            return None


# --- LLM response cache ----------------------------------------------------


def cache_get(
    cache_key: str, path: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    init_db(path)
    with connect(path) as conn:
        row = conn.execute(
            "SELECT response_json FROM llm_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if row is None:
            return None
        try:
            return json.loads(row["response_json"])
        except Exception:
            return None


def cache_set(
    cache_key: str,
    model: str,
    agent_name: str,
    response: Dict[str, Any],
    path: Optional[str] = None,
) -> None:
    init_db(path)
    with connect(path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO llm_cache
                (cache_key, created_at, model, agent_name, response_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                cache_key,
                utcnow_iso(),
                model,
                agent_name,
                json.dumps(response),
            ),
        )
