"""Cost-control helpers: compaction, evidence compression, early stopping.

These functions are pure and easy to test. They keep the agent inputs
short, which is the single biggest cost lever in a multi-round
multi-agent system.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from . import config as _config_mod
from .schemas import (
    AgentOutput,
    DiscussionSummary,
    EvidenceChunk,
    EvidenceSummary,
)
from .utils import truncate


def compact_evidence_for_agents(
    evidence: EvidenceSummary,
    max_chars: Optional[int] = None,
) -> str:
    """One short paragraph all domain agents share - never raw chunks."""
    max_chars = max_chars or _config_mod.CONFIG.max_evidence_chars
    parts: List[str] = []
    if evidence.observed_facts:
        parts.append("Observed: " + "; ".join(evidence.observed_facts[:3]))
    if evidence.historical_analogies:
        parts.append("History: " + "; ".join(evidence.historical_analogies[:3]))
    if evidence.strategy_frameworks:
        parts.append(
            "Frameworks: " + "; ".join(evidence.strategy_frameworks[:3])
        )
    if evidence.hypothetical_assumptions:
        parts.append(
            "Hypothetical: " + "; ".join(evidence.hypothetical_assumptions[:3])
        )
    if evidence.compact_summary:
        parts.append("Summary: " + evidence.compact_summary)
    if evidence.note:
        parts.append("Note: " + evidence.note)
    return truncate(" | ".join(parts), max_chars)


def compact_agent_position(output: AgentOutput, max_chars: int = 350) -> str:
    """One-line position summary for the next round's prompt."""
    pieces = [output.main_assessment]
    if output.key_drivers:
        pieces.append("Drivers: " + "; ".join(output.key_drivers[:3]))
    if output.disagreements:
        pieces.append("Disagrees on: " + "; ".join(output.disagreements[:2]))
    return truncate(" / ".join(p for p in pieces if p), max_chars)


def build_discussion_summary(
    round_number: int,
    latest_outputs: Dict[str, AgentOutput],
) -> DiscussionSummary:
    """Heuristic compaction used in mock mode AND as a fallback.

    The Orchestrator may also produce its own LLM-generated summary; this
    one is the deterministic safety net.
    """
    agreements: List[str] = []
    disagreements: List[str] = []
    uncertainties: List[str] = []
    timeline_bits: List[str] = []
    positions: Dict[str, str] = {}

    for name, out in latest_outputs.items():
        positions[name] = compact_agent_position(out)
        agreements.extend(out.agreements[:2])
        disagreements.extend(out.disagreements[:2])
        uncertainties.extend(out.uncertainties[:2])
        for tc in out.timeline_contributions[:2]:
            timeline_bits.append(
                "{y}: {e}".format(y=tc.year, e=truncate(tc.event, 80))
            )

    def _dedupe(items: List[str], cap: int) -> List[str]:
        seen: List[str] = []
        for x in items:
            x = (x or "").strip()
            if not x:
                continue
            if x.lower() in (s.lower() for s in seen):
                continue
            seen.append(x)
            if len(seen) >= cap:
                break
        return seen

    deduped_disagreements = _dedupe(disagreements, 6)
    query_terms: List[str] = []
    for d in deduped_disagreements:
        for word in d.split():
            w = word.strip(".,;:").lower()
            if len(w) > 4 and w not in query_terms:
                query_terms.append(w)
            if len(query_terms) >= 8:
                break
        if len(query_terms) >= 8:
            break

    return DiscussionSummary(
        round_number=round_number,
        areas_of_agreement=_dedupe(agreements, 6),
        areas_of_disagreement=deduped_disagreements,
        emerging_timeline=_dedupe(timeline_bits, 8),
        key_uncertainties=_dedupe(uncertainties, 6),
        agent_positions=positions,
        disagreement_query_terms=query_terms,
    )


def should_stop_early(
    round_number: int,
    summary: DiscussionSummary,
    min_round: int = 2,
) -> bool:
    """Stop after round >= `min_round` when consensus looks solid.

    The graph still supports up to MAX_AGENT_DISCUSSION_ROUNDS; this is
    an optimization, not a hard rule.
    """
    if round_number < min_round:
        return False
    if len(summary.areas_of_disagreement) <= 1 and len(summary.emerging_timeline) >= 3:
        return True
    return False
