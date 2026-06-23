"""LangGraph checkpoint persistence (SQLite) for crash recovery."""
from __future__ import annotations

import os
import sqlite3
from contextvars import ContextVar, Token
from typing import Any, Dict, Optional

from . import config as _config_mod
from .llm import LLMClient

_active_llm: ContextVar[Optional[LLMClient]] = ContextVar("_active_llm", default=None)
_checkpointer_cache: Dict[str, Any] = {}


def graph_run_config(run_id: str) -> Dict[str, Any]:
    return {"configurable": {"thread_id": run_id}}


def set_active_llm(llm: LLMClient) -> Token:
    return _active_llm.set(llm)


def reset_active_llm(token: Token) -> None:
    _active_llm.reset(token)


def get_active_llm() -> Optional[LLMClient]:
    return _active_llm.get()


def reset_checkpointer_cache() -> None:
    for saver in _checkpointer_cache.values():
        try:
            saver.conn.close()
        except Exception:
            pass
    _checkpointer_cache.clear()


def get_checkpointer(path: Optional[str] = None):
    """Return a shared SqliteSaver, or None when checkpointing is disabled."""
    if not _config_mod.CONFIG.enable_graph_checkpoints:
        return None
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except Exception:
        return None

    db_path = path or _config_mod.CONFIG.checkpoint_sqlite_path
    if db_path in _checkpointer_cache:
        return _checkpointer_cache[db_path]

    parent = os.path.dirname(db_path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

    conn = sqlite3.connect(db_path, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    _checkpointer_cache[db_path] = saver
    return saver


def get_checkpoint_status(run_id: str) -> Optional[Dict[str, Any]]:
    """Return resume metadata for a run, or None if no checkpoint exists."""
    from .graph import build_graph

    checkpointer = get_checkpointer()
    if checkpointer is None:
        return None

    compiled = build_graph(checkpointer=checkpointer)
    if compiled is None:
        return None

    config = graph_run_config(run_id)
    snapshot = compiled.get_state(config)
    if not snapshot.values:
        return None

    values = snapshot.values
    metrics = values.get("run_metrics") or {}
    next_nodes = list(snapshot.next or ())
    return {
        "run_id": run_id,
        "can_resume": bool(next_nodes),
        "next_nodes": next_nodes,
        "seed": values.get("seed"),
        "scenario_mode": values.get("scenario_mode"),
        "scenario_title": values.get("scenario_title") or "",
        "discussion_rounds_completed": int(metrics.get("discussion_rounds_completed") or 0),
    }
