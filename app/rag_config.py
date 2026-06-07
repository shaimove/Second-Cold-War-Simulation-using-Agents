"""Tier 2 RAG configuration (env-overridable)."""
from __future__ import annotations

import os
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


RAG_CONFIG: Dict[str, Any] = {
    "baseline_candidate_k": 25,
    "baseline_final_k": 8,
    "agent_round1_candidate_k": 15,
    "agent_round1_final_k": 4,
    "disagreement_candidate_k": 15,
    "disagreement_final_k": 4,
    "red_team_candidate_k": 15,
    "red_team_final_k": 4,
    "max_chunk_chars": 900,
    "max_lane_chars": 1200,
    "max_final_evidence_items": 12,
    "enable_agent_rag": True,
    "enable_disagreement_rag": True,
    "enable_red_team_rag": True,
    "enable_round3_new_retrieval": False,
    "max_chunks_per_source": 2,
}


_ENV_MAP = {
    "RAG_BASELINE_CANDIDATE_K": "baseline_candidate_k",
    "RAG_BASELINE_FINAL_K": "baseline_final_k",
    "RAG_AGENT_ROUND1_CANDIDATE_K": "agent_round1_candidate_k",
    "RAG_AGENT_ROUND1_FINAL_K": "agent_round1_final_k",
    "RAG_DISAGREEMENT_CANDIDATE_K": "disagreement_candidate_k",
    "RAG_DISAGREEMENT_FINAL_K": "disagreement_final_k",
    "RAG_RED_TEAM_CANDIDATE_K": "red_team_candidate_k",
    "RAG_RED_TEAM_FINAL_K": "red_team_final_k",
    "RAG_MAX_CHUNK_CHARS": "max_chunk_chars",
    "RAG_MAX_LANE_CHARS": "max_lane_chars",
    "RAG_MAX_FINAL_EVIDENCE_ITEMS": "max_final_evidence_items",
    "RAG_ENABLE_AGENT_RAG": "enable_agent_rag",
    "RAG_ENABLE_DISAGREEMENT_RAG": "enable_disagreement_rag",
    "RAG_ENABLE_RED_TEAM_RAG": "enable_red_team_rag",
    "RAG_ENABLE_ROUND3_NEW_RETRIEVAL": "enable_round3_new_retrieval",
}


def load_rag_config() -> Dict[str, Any]:
    """Return a copy of RAG_CONFIG with environment overrides applied."""
    cfg = dict(RAG_CONFIG)
    for env_name, key in _ENV_MAP.items():
        if key.startswith("enable_"):
            cfg[key] = _get_bool(env_name, bool(cfg[key]))
        else:
            cfg[key] = _get_int(env_name, int(cfg[key]))
    return cfg


# Module-level config used by rag.py
ACTIVE_RAG_CONFIG = load_rag_config()
