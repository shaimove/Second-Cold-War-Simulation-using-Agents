"""Shared test fixtures.

Every test runs against an isolated temp SQLite DB and forces mock mode
(no API key). This guarantees zero network calls and deterministic
behavior across the whole suite.
"""
from __future__ import annotations

import os
import sys

import pytest


# Ensure project root is importable.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch, tmp_path):
    """Wipe API key and point all paths to a temp directory."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setenv("OPENAI_IMAGE_MODEL", "test-image")
    monkeypatch.setenv("USE_RAG", "true")
    monkeypatch.setenv("USE_LLM_CACHE", "true")
    monkeypatch.setenv("ENABLE_IMAGE_GENERATION", "true")
    monkeypatch.setenv("PARALLEL_DOMAIN_AGENTS", "true")
    monkeypatch.setenv("MAX_AGENT_DISCUSSION_ROUNDS", "3")
    monkeypatch.setenv("MAX_RETRIEVED_DOCS", "3")
    monkeypatch.setenv("SELF_POSITION_MAX_CHARS", "1000")
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "test.sqlite"))
    monkeypatch.setenv(
        "RAG_CHUNKS_PATH", str(tmp_path / "rag_chunks.json")
    )
    monkeypatch.setenv(
        "GENERATED_IMAGES_DIR", str(tmp_path / "generated_images")
    )
    monkeypatch.setenv(
        "CHECKPOINT_SQLITE_PATH", str(tmp_path / "checkpoints.sqlite")
    )
    monkeypatch.setenv("ENABLE_GRAPH_CHECKPOINTS", "true")

    # Rebuild config + db modules so they pick up the new env.
    from app import config as cfg_mod
    cfg_mod.CONFIG = cfg_mod.load_config()

    from app import db as db_mod
    db_mod.init_db()

    from app.checkpoints import reset_checkpointer_cache
    reset_checkpointer_cache()

    # Reset retrieval cache between tests.
    from app import rag as rag_mod
    rag_mod._RETRIEVAL_CACHE.clear()

    yield
