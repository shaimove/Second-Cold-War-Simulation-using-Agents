"""Application configuration.

All runtime config is read from environment variables (.env). The
`Config` object exposes typed accessors so the rest of the app never
reads `os.environ` directly. This makes the model, feature flags, and
paths easy to swap without code changes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

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


@dataclass(frozen=True)
class Config:
    openai_api_key: Optional[str]
    openai_model: str
    openai_orchestrator_model: str
    openai_judge_model: str
    openai_image_model: str

    use_rag: bool
    use_llm_cache: bool
    enable_image_generation: bool
    enable_run_judge: bool
    enable_year_judge: bool
    enable_timeline_judge: bool
    parallel_domain_agents: bool

    max_agent_discussion_rounds: int
    max_retrieved_docs: int
    max_agent_input_chars: int
    max_evidence_chars: int
    self_position_max_chars: int

    enable_graph_checkpoints: bool
    sqlite_path: str
    checkpoint_sqlite_path: str
    rag_chunks_path: str
    rag_chroma_path: str
    rag_embedding_model: str
    generated_images_dir: str

    @property
    def mock_mode(self) -> bool:
        """True when no API key is available; the app then uses stubs."""
        return not bool(self.openai_api_key)


def load_config() -> Config:
    return Config(
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        openai_model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
        openai_orchestrator_model=os.getenv(
            "OPENAI_ORCHESTRATOR_MODEL", "gpt-5.4"
        ),
        openai_judge_model=os.getenv(
            "OPENAI_JUDGE_MODEL",
            os.getenv("OPENAI_ORCHESTRATOR_MODEL", "gpt-5.4"),
        ),
        openai_image_model=os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2"),
        use_rag=_get_bool("USE_RAG", True),
        use_llm_cache=_get_bool("USE_LLM_CACHE", True),
        enable_image_generation=_get_bool("ENABLE_IMAGE_GENERATION", True),
        enable_run_judge=_get_bool("ENABLE_RUN_JUDGE", False),
        enable_year_judge=_get_bool("ENABLE_YEAR_JUDGE", True),
        enable_timeline_judge=_get_bool("ENABLE_TIMELINE_JUDGE", True),
        parallel_domain_agents=_get_bool("PARALLEL_DOMAIN_AGENTS", True),
        max_agent_discussion_rounds=_get_int("MAX_AGENT_DISCUSSION_ROUNDS", 3),
        max_retrieved_docs=_get_int("MAX_RETRIEVED_DOCS", 5),
        max_agent_input_chars=_get_int("MAX_AGENT_INPUT_CHARS", 8000),
        max_evidence_chars=_get_int("MAX_EVIDENCE_CHARS", 2500),
        self_position_max_chars=_get_int("SELF_POSITION_MAX_CHARS", 1000),
        enable_graph_checkpoints=_get_bool("ENABLE_GRAPH_CHECKPOINTS", True),
        sqlite_path=os.getenv("SQLITE_PATH", "data/scenarios.sqlite"),
        checkpoint_sqlite_path=os.getenv(
            "CHECKPOINT_SQLITE_PATH", "data/checkpoints.sqlite"
        ),
        rag_chunks_path=os.getenv("RAG_CHUNKS_PATH", "data/rag_chunks.json"),
        rag_chroma_path=os.getenv("RAG_CHROMA_PATH", "data/chroma"),
        rag_embedding_model=os.getenv(
            "RAG_EMBEDDING_MODEL", "text-embedding-3-small"
        ),
        generated_images_dir=os.getenv(
            "GENERATED_IMAGES_DIR", "data/generated_images"
        ),
    )


CONFIG = load_config()
