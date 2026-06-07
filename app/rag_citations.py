"""Evidence packets per agent, citation validation, prompt formatting."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .rag_config import ACTIVE_RAG_CONFIG
from .schemas import (
    AgentOutput,
    DiscussionSummary,
    EvidenceChunk,
    EvidenceLanes,
    GroundingNote,
    RAG_INFLUENCE_VALUES,
)
from .utils import truncate


AGENT_LANE_KEYS: Dict[str, List[str]] = {
    "geo_strategy": ["observed_blob", "geostrategy_blob", "frameworks_blob", "general_blob"],
    "economy_technology": ["observed_blob", "economy_blob", "frameworks_blob"],
    "domestic_ideology": ["observed_blob", "domestic_blob", "frameworks_blob"],
    "security_taiwan": ["observed_blob", "security_blob", "frameworks_blob"],
    "historical_analogy": ["historical_blob", "frameworks_blob"],
}


def register_chunks(
    registry: Dict[str, EvidenceChunk],
    chunks: List[EvidenceChunk],
) -> None:
    for ch in chunks:
        if isinstance(ch, dict):
            ch = EvidenceChunk(**ch)
        if ch.chunk_id:
            registry[ch.chunk_id] = ch


def format_chunks_block(chunks: List[EvidenceChunk], title: str) -> str:
    if not chunks:
        return ""
    lines = [title + ":"]
    for ch in chunks:
        lines.append(
            "[{id}] ({dom}/{st}) {src}\n{txt}".format(
                id=ch.chunk_id,
                dom=ch.domain,
                st=ch.source_type,
                src=ch.source_name or ch.source_path,
                txt=ch.text,
            )
        )
    return "\n".join(lines)


def lanes_text_for_agent(agent_name: str, lanes: EvidenceLanes) -> str:
    keys = AGENT_LANE_KEYS.get(agent_name, ["observed_blob", "general_blob"])
    parts: List[str] = []
    for key in keys:
        val = getattr(lanes, key, "") or ""
        if val:
            parts.append(key.replace("_blob", "").upper() + ":\n" + val)
    return "\n\n".join(parts)


def build_agent_evidence_packet(
    agent_name: str,
    lanes: EvidenceLanes,
    agent_chunks: List[EvidenceChunk],
    disagreement_chunks: Optional[List[EvidenceChunk]] = None,
    round_number: int = 1,
) -> Tuple[str, List[str]]:
    """Return (prompt text, chunk_ids provided to agent)."""
    parts: List[str] = []
    lane_text = lanes_text_for_agent(agent_name, lanes)
    if lane_text:
        parts.append("Evidence lanes:\n" + lane_text)

    if agent_chunks:
        parts.append(format_chunks_block(agent_chunks, "Agent-specific retrieved chunks"))

    if disagreement_chunks and round_number >= 2:
        parts.append(format_chunks_block(disagreement_chunks, "Shared disagreement evidence"))

    allowed_ids = []
    for ch in (agent_chunks or []) + (disagreement_chunks or []):
        if ch.chunk_id:
            allowed_ids.append(ch.chunk_id)

    max_chars = int(ACTIVE_RAG_CONFIG.get("max_lane_chars", 1200)) * 3
    return truncate("\n\n".join(parts), max_chars), allowed_ids


CITATION_SCHEMA_APPENDIX = """
Also include RAG citation fields (only cite chunk_ids from RETRIEVED CHUNKS above):
{
  "sources_used": ["kb_000001"],
  "grounding_notes": [{"chunk_id": "kb_000001", "claim": "one sentence"}],
  "rag_influence": "changed_view|supported_view|contradicted_view|not_used",
  "rag_influence_explanation": "one short sentence"
}
"""


def validate_and_apply_citations(
    data: Dict[str, Any],
    allowed_chunk_ids: List[str],
    agent_name: str,
    recorder: Optional[Any] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """Strip invalid citations; return warnings."""
    allowed = set(allowed_chunk_ids or [])
    warnings: List[str] = []

    sources = data.get("sources_used") or []
    if isinstance(sources, str):
        sources = [sources]
    valid_sources: List[str] = []
    if isinstance(sources, list):
        for cid in sources:
            cid = str(cid).strip()
            if not cid:
                continue
            if allowed and cid not in allowed:
                msg = f"{agent_name}: removed invalid chunk_id {cid}"
                warnings.append(msg)
                if recorder:
                    recorder.add_warning(msg)
                continue
            valid_sources.append(cid)

    notes_raw = data.get("grounding_notes") or []
    valid_notes: List[Dict[str, str]] = []
    if isinstance(notes_raw, list):
        for raw in notes_raw:
            if not isinstance(raw, dict):
                continue
            cid = str(raw.get("chunk_id") or "").strip()
            claim = str(raw.get("claim") or "")[:400]
            if not cid:
                continue
            if allowed and cid not in allowed:
                msg = f"{agent_name}: removed grounding_note for unknown chunk_id {cid}"
                warnings.append(msg)
                if recorder:
                    recorder.add_warning(msg)
                continue
            valid_notes.append({"chunk_id": cid, "claim": claim})

    influence = str(data.get("rag_influence") or "not_used").strip().lower()
    if influence not in RAG_INFLUENCE_VALUES:
        influence = "not_used"

    data = dict(data)
    data["sources_used"] = valid_sources
    data["grounding_notes"] = valid_notes
    data["rag_influence"] = influence
    data["rag_influence_explanation"] = str(data.get("rag_influence_explanation") or "")[:400]

    if recorder and valid_sources:
        recorder.record_citations(agent_name, valid_sources)

    return data, warnings


def apply_citations_to_output(
    output: AgentOutput,
    data: Dict[str, Any],
) -> AgentOutput:
    output.sources_used = list(data.get("sources_used") or [])
    notes: List[GroundingNote] = []
    for raw in data.get("grounding_notes") or []:
        if isinstance(raw, dict):
            notes.append(
                GroundingNote(
                    chunk_id=str(raw.get("chunk_id") or ""),
                    claim=str(raw.get("claim") or "")[:400],
                )
            )
    output.grounding_notes = notes
    output.rag_influence = str(data.get("rag_influence") or "not_used")
    output.rag_influence_explanation = str(data.get("rag_influence_explanation") or "")[:400]
    return output
