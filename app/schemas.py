"""Pydantic schemas for state and final output.

These types define the contract between agents, the graph, and the API.
They are intentionally permissive (most fields default-empty) so the
mock pipeline and partial failures still produce a serializable result.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


SCENARIO_MODES = ("base_case", "escalation", "de_escalation", "wildcard")
EVENT_STATUSES = ("observed", "hypothetical", "mixed")
DOMAINS = (
    "economy",
    "strategy",
    "technology",
    "security",
    "ideology",
    "historical",
)
IMPACT_LEVELS = ("low", "medium", "high")
CONFIDENCE_LEVELS = ("low", "medium", "high")
AGENT_NAMES = (
    "orchestrator",
    "evidence_rag",
    "geo_strategy",
    "economy_technology",
    "domestic_ideology",
    "security_taiwan",
    "historical_analogy",
    "red_team",
)
DOMAIN_AGENTS = (
    "geo_strategy",
    "economy_technology",
    "domestic_ideology",
    "security_taiwan",
    "historical_analogy",
)


RAG_DOMAINS = (
    "economy_technology",
    "security_taiwan",
    "geo_strategy",
    "domestic_ideology",
    "historical_analogy",
    "strategy_framework",
    "general",
    "unknown",
)
RAG_SOURCE_TYPES = (
    "book",
    "report",
    "framework",
    "current_context",
    "historical_analogy",
    "strategy_framework",
    "unknown",
)
RAG_PERIODS = (
    "modern_us_china",
    "us_ussr_cold_war",
    "general",
    "unknown",
)
RAG_INFLUENCE_VALUES = (
    "changed_view",
    "supported_view",
    "contradicted_view",
    "not_used",
)


class EvidenceChunk(BaseModel):
    chunk_id: str = ""
    source_path: str = ""
    source_name: str = ""
    domain: str = "general"
    source_type: str = "unknown"
    period: str = "unknown"
    text: str = ""
    score: float = 0.0
    char_count: int = 0


class EvidenceSummary(BaseModel):
    observed_facts: List[str] = Field(default_factory=list)
    historical_analogies: List[str] = Field(default_factory=list)
    strategy_frameworks: List[str] = Field(default_factory=list)
    hypothetical_assumptions: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    compact_summary: str = ""
    note: str = ""


class EvidenceLanes(BaseModel):
    observed_blob: str = ""
    historical_blob: str = ""
    frameworks_blob: str = ""
    economy_blob: str = ""
    security_blob: str = ""
    domestic_blob: str = ""
    geostrategy_blob: str = ""
    general_blob: str = ""


class GroundingNote(BaseModel):
    chunk_id: str = ""
    claim: str = ""


class FinalEvidencePacket(BaseModel):
    """Compact evidence for final orchestrator synthesis."""

    items: List[str] = Field(default_factory=list)
    source_files: List[str] = Field(default_factory=list)
    chunk_ids: List[str] = Field(default_factory=list)
    text: str = ""


class TimelineEvent(BaseModel):
    event: str = ""
    domain: str = "strategy"
    probability: float = 0.5
    impact: str = "medium"
    confidence: str = "medium"
    rationale: str = ""

    @field_validator("probability")
    @classmethod
    def _clamp_probability(cls, v: float) -> float:
        try:
            v = float(v)
        except Exception:
            return 0.5
        if v < 0:
            return 0.0
        if v > 1:
            return 1.0
        return v


class YearBlock(BaseModel):
    year: int
    headline: str = ""
    events: List[TimelineEvent] = Field(default_factory=list)


class AgentTimelineContribution(BaseModel):
    year: int
    event: str = ""
    probability: float = 0.5
    impact: str = "medium"
    confidence: str = "medium"
    rationale: str = ""


class AgentOutput(BaseModel):
    agent_name: str
    round_number: int = 1
    main_assessment: str = ""
    key_drivers: List[str] = Field(default_factory=list)
    timeline_contributions: List[AgentTimelineContribution] = Field(
        default_factory=list
    )
    risks: List[str] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)
    agreements: List[str] = Field(default_factory=list)
    disagreements: List[str] = Field(default_factory=list)
    position_changed_from_previous_round: bool = False
    sources_used: List[str] = Field(default_factory=list)
    grounding_notes: List[GroundingNote] = Field(default_factory=list)
    rag_influence: str = "not_used"
    rag_influence_explanation: str = ""


class DiscussionSummary(BaseModel):
    round_number: int
    areas_of_agreement: List[str] = Field(default_factory=list)
    areas_of_disagreement: List[str] = Field(default_factory=list)
    emerging_timeline: List[str] = Field(default_factory=list)
    key_uncertainties: List[str] = Field(default_factory=list)
    agent_positions: Dict[str, str] = Field(default_factory=dict)
    disagreement_query_terms: List[str] = Field(default_factory=list)


class RedTeamFinding(BaseModel):
    issue: str
    severity: str = "medium"
    affected_assumption: str = ""


class RunMetrics(BaseModel):
    llm_calls: int = 0
    agents_used: List[str] = Field(default_factory=list)
    retrieved_docs: int = 0
    cache_hits: int = 0
    rag_calls: int = 0
    rag_cache_hits: int = 0
    retrieved_candidate_chunks: int = 0
    retrieved_final_chunks: int = 0
    unique_chunks_used: int = 0
    most_used_source_files: List[str] = Field(default_factory=list)
    rag_source_files: List[str] = Field(default_factory=list)
    retrieved_chunk_ids: List[str] = Field(default_factory=list)
    most_cited_chunk_ids: List[str] = Field(default_factory=list)
    per_agent_sources_used: Dict[str, List[str]] = Field(default_factory=dict)
    citation_warnings: List[str] = Field(default_factory=list)
    discussion_rounds_completed: int = 0
    elapsed_seconds: float = 0.0
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    # Final orchestrator synthesis validation / repair (only this agent)
    synthesis_validation_passed: bool = False
    synthesis_repair_attempts: int = 0
    synthesis_regeneration_attempts: int = 0
    synthesis_validation_errors: List[str] = Field(default_factory=list)
    synthesis_used_fallback: bool = False
    synthesis_error_message: Optional[str] = None


class OrchestratorSynthesisOutput(BaseModel):
    """Validated JSON contract for the final orchestrator LLM call."""

    scenario_title: str = Field(..., min_length=1, max_length=200)
    scenario_summary: str = Field(..., min_length=1, max_length=1200)
    event_status: str = "hypothetical"
    key_assumptions: List[str] = Field(default_factory=list)
    main_disagreements: List[str] = Field(default_factory=list)
    image_prompt: str = Field(default="", max_length=1200)

    @field_validator("event_status")
    @classmethod
    def _normalize_event_status(cls, v: str) -> str:
        v = (v or "hypothetical").strip().lower()
        if v not in EVENT_STATUSES:
            return "hypothetical"
        return v

    @field_validator("key_assumptions", "main_disagreements")
    @classmethod
    def _coerce_str_lists(cls, v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [v.strip()] if v.strip() else []
        if isinstance(v, list):
            out: List[str] = []
            for item in v:
                if item is None:
                    continue
                if isinstance(item, str) and item.strip():
                    out.append(item.strip()[:400])
                elif isinstance(item, (int, float)):
                    out.append(str(item)[:400])
            return out
        return []

    @field_validator("scenario_title", "scenario_summary", "image_prompt")
    @classmethod
    def _strip_strings(cls, v: str) -> str:
        return (v or "").strip()


class ImageResult(BaseModel):
    enabled: bool = False
    generated: bool = False
    path: Optional[str] = None
    error: Optional[str] = None
    mock: bool = False


class ScenarioState(BaseModel):
    """Mutable state passed between LangGraph nodes."""

    run_id: str
    seed: str
    scenario_mode: str = "base_case"
    event_status: str = "hypothetical"
    current_year: int = 2026
    simulation_years: List[int] = Field(
        default_factory=lambda: [2026, 2027, 2028, 2029, 2030, 2031]
    )
    evidence_summary: EvidenceSummary = Field(default_factory=EvidenceSummary)
    evidence_lanes: EvidenceLanes = Field(default_factory=EvidenceLanes)
    baseline_chunks: List[EvidenceChunk] = Field(default_factory=list)
    disagreement_chunks: List[EvidenceChunk] = Field(default_factory=list)
    red_team_chunks: List[EvidenceChunk] = Field(default_factory=list)
    final_evidence_packet: FinalEvidencePacket = Field(
        default_factory=FinalEvidencePacket
    )
    discussion_rounds: List[DiscussionSummary] = Field(default_factory=list)
    agent_outputs: Dict[str, List[AgentOutput]] = Field(default_factory=dict)
    disagreements: List[str] = Field(default_factory=list)
    red_team_findings: List[RedTeamFinding] = Field(default_factory=list)
    final_timeline: List[YearBlock] = Field(default_factory=list)
    scenario_title: str = ""
    scenario_summary: str = ""
    image_prompt: str = ""
    image_result: ImageResult = Field(default_factory=ImageResult)
    run_metrics: RunMetrics = Field(default_factory=RunMetrics)
    errors: List[str] = Field(default_factory=list)
    chunks_used_registry: Dict[str, EvidenceChunk] = Field(default_factory=dict)

    @field_validator("scenario_mode")
    @classmethod
    def _validate_mode(cls, v: str) -> str:
        if v not in SCENARIO_MODES:
            raise ValueError(
                "scenario_mode must be one of: " + ", ".join(SCENARIO_MODES)
            )
        return v

    @field_validator("event_status")
    @classmethod
    def _validate_status(cls, v: str) -> str:
        if v not in EVENT_STATUSES:
            return "hypothetical"
        return v


class ScenarioRequest(BaseModel):
    seed: str
    scenario_mode: str = "base_case"

    @field_validator("scenario_mode")
    @classmethod
    def _validate_mode(cls, v: str) -> str:
        if v not in SCENARIO_MODES:
            raise ValueError(
                "scenario_mode must be one of: " + ", ".join(SCENARIO_MODES)
            )
        return v


class FinalScenario(BaseModel):
    """Public-facing output returned to the frontend and saved to DB."""

    run_id: str
    scenario_title: str = ""
    scenario_summary: str = ""
    seed: str
    scenario_mode: str
    event_status: str = "hypothetical"
    timeline: List[YearBlock] = Field(default_factory=list)
    key_assumptions: List[str] = Field(default_factory=list)
    main_disagreements: List[str] = Field(default_factory=list)
    red_team_warnings: List[str] = Field(default_factory=list)
    agent_summaries: Dict[str, str] = Field(default_factory=dict)
    discussion_summary: List[DiscussionSummary] = Field(default_factory=list)
    image_prompt: str = ""
    image: ImageResult = Field(default_factory=ImageResult)
    run_metrics: RunMetrics = Field(default_factory=RunMetrics)


class SavedRunSummary(BaseModel):
    run_id: str
    created_at: str
    seed: str
    scenario_mode: str
    scenario_title: str


def empty_final_scenario(
    run_id: str, seed: str, scenario_mode: str
) -> Dict[str, Any]:
    """A safe default shape used by error paths and fallbacks."""
    return FinalScenario(
        run_id=run_id, seed=seed, scenario_mode=scenario_mode
    ).model_dump()
