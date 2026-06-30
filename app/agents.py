"""All agents.

Each agent is a function that:
- builds a small, schema-constrained prompt
- calls `LLMClient.call_llm_json` (or text where appropriate)
- coerces the response into a Pydantic model

Agents never read OpenAI, env, or DB directly - they go through
`LLMClient` so the same code runs in real mode, mock mode, and tests.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from . import config as _config_mod
from .cost_control import compact_evidence_for_agents
from .llm import LLMClient
from . import prompts as prompt_mod
from .rag_citations import (
    CITATION_SCHEMA_APPENDIX,
    apply_citations_to_output,
    build_agent_evidence_packet,
    validate_and_apply_citations,
)
from .schemas import (
    AgentOutput,
    AgentTimelineContribution,
    DiscussionSummary,
    EvidenceChunk,
    EvidenceSummary,
    FinalEvidencePacket,
    RedTeamFinding,
    TimelineEvent,
    YearBlock,
)
from .utils import SIMULATION_YEARS, assemble_user_prompt, budget_prompt_sections, truncate


def format_resolved_timeline(blocks: List[YearBlock]) -> str:
    """Compact text of locked prior years for agent prompts."""
    if not blocks:
        return "<none — this is the first simulation year>"
    lines: List[str] = []
    for block in blocks:
        lines.append("{y}: {h}".format(y=block.year, h=block.headline or "(no headline)"))
        for event in block.events[:4]:
            if event.event:
                lines.append("  - " + truncate(event.event, 120))
    return truncate("\n".join(lines), 2000)


# --- Evidence agent --------------------------------------------------------


def run_evidence_agent(
    llm: LLMClient,
    seed: str,
    scenario_mode: str,
    baseline_chunks: Optional[List[EvidenceChunk]] = None,
) -> tuple:
    """Returns (EvidenceSummary, retrieved_docs_count, cache_hit)."""
    chunks: List[EvidenceChunk] = list(baseline_chunks or [])
    cache_hit = False

    chunk_blob = "\n\n".join(
        "[{id}] SOURCE: {p} ({t}/{d}/{per})\n{txt}".format(
            id=c.chunk_id,
            p=c.source_path,
            t=c.source_type,
            d=c.domain,
            per=c.period,
            txt=c.text,
        )
        for c in chunks
    )
    sources = sorted({c.source_path for c in chunks if c.source_path})
    chunk_ids = [c.chunk_id for c in chunks if c.chunk_id]

    system = prompt_mod.EVIDENCE_AGENT_SYSTEM + prompt_mod.SAFETY_TAIL + prompt_mod.JSON_TAIL

    user = (
        "Seed: " + seed + "\n"
        "Scenario mode: " + scenario_mode + "\n\n"
        "Retrieved context (may be empty):\n" + (chunk_blob or "<none>") + "\n\n"
        "Required JSON shape:\n"
        "{\n"
        '  "observed_facts": ["..."],\n'
        '  "historical_analogies": ["..."],\n'
        '  "strategy_frameworks": ["..."],\n'
        '  "hypothetical_assumptions": ["..."],\n'
        '  "compact_summary": "<<=400 chars>",\n'
        '  "note": "<short caveat about retrieval coverage>"\n'
        "}\n"
    )

    data = llm.call_llm_json(
        system_prompt=system,
        user_prompt=user,
        agent_name="evidence_rag",
        round_number=0,
        schema_name="evidence_summary",
        cache_context={
            "seed": seed,
            "mode": scenario_mode,
            "n_chunks": len(chunks),
            "chunk_ids": chunk_ids,
        },
        fallback={
            "observed_facts": [],
            "historical_analogies": [],
            "strategy_frameworks": [],
            "hypothetical_assumptions": [
                "User seed treated as a hypothetical future event."
            ],
            "compact_summary": truncate(
                "Seed: " + seed + " | Mode: " + scenario_mode, 400
            ),
            "note": "Evidence agent fallback used.",
        },
    )

    summary = EvidenceSummary(
        observed_facts=_as_str_list(data.get("observed_facts")),
        historical_analogies=_as_str_list(data.get("historical_analogies")),
        strategy_frameworks=_as_str_list(data.get("strategy_frameworks")),
        hypothetical_assumptions=_as_str_list(data.get("hypothetical_assumptions"))
        or ["User seed treated as a hypothetical future event."],
        sources=sources,
        compact_summary=str(data.get("compact_summary") or "")[:600],
        note=str(data.get("note") or "")[:300],
    )
    return summary, len(chunks), False


# --- Domain agents ---------------------------------------------------------


def run_domain_agent(
    llm: LLMClient,
    agent_name: str,
    seed: str,
    scenario_mode: str,
    target_year: int,
    resolved_timeline: Optional[List[YearBlock]] = None,
    evidence_packet: str = "",
    round_number: int = 1,
    previous_summary: Optional[DiscussionSummary] = None,
    previous_self_position: Optional[str] = None,
    allowed_chunk_ids: Optional[List[str]] = None,
    rag_recorder: Optional[Any] = None,
    *,
    evidence_blob: Optional[str] = None,
) -> AgentOutput:
    if agent_name not in prompt_mod.AGENT_SYSTEM_PROMPTS:
        raise ValueError("Unknown agent: " + agent_name)

    system = (
        prompt_mod.AGENT_SYSTEM_PROMPTS[agent_name]
        + prompt_mod.DOMAIN_AGENT_YEAR_FOCUS
        + prompt_mod.SAFETY_TAIL
        + "\n\nWrite as an expert analyst. Be concise."
        + prompt_mod.JSON_TAIL
    )

    prev_summary_text = ""
    if previous_summary is not None:
        prev_summary_text = json.dumps(
            previous_summary.model_dump(), ensure_ascii=False
        )

    packet = evidence_packet or evidence_blob or ""
    prior_history = format_resolved_timeline(resolved_timeline or [])
    fixed_suffix = (
        prompt_mod.domain_agent_schema_hint(target_year) + CITATION_SCHEMA_APPENDIX
    )
    max_chars = _config_mod.CONFIG.max_agent_input_chars
    context = budget_prompt_sections(
        [
            (
                "Seed: " + seed + "\n"
                "Scenario mode: " + scenario_mode + "\n"
                "Target simulation year: " + str(target_year),
                1,
            ),
            (
                "Locked prior years (established scenario history):\n" + prior_history,
                2,
            ),
            (
                "Evidence packet (lanes + retrieved chunks):\n" + packet,
                5,
            ),
            (
                "Discussion round for "
                + str(target_year)
                + ": "
                + str(round_number)
                + "\n"
                "Previous-round discussion summary:\n"
                + (prev_summary_text or "<none - this is round 1>"),
                4,
            ),
            (
                "Your previous position for this year (if any):\n"
                + (previous_self_position or "<none>"),
                3,
            ),
        ],
        max_chars - len(fixed_suffix),
    )
    user = context + fixed_suffix

    fallback = _domain_fallback(agent_name, round_number, target_year)
    data = llm.call_llm_json(
        system_prompt=system,
        user_prompt=user,
        agent_name=agent_name,
        round_number=round_number,
        cache_context={
            "seed": seed,
            "mode": scenario_mode,
            "year": target_year,
            "round": round_number,
            "prev_pos": previous_self_position or "",
        },
        fallback=fallback,
    )
    data, _warnings = validate_and_apply_citations(
        data, allowed_chunk_ids or [], agent_name, rag_recorder
    )
    return _to_agent_output(agent_name, round_number, target_year, data)


# --- Red-Team agent --------------------------------------------------------


def run_red_team_agent(
    llm: LLMClient,
    seed: str,
    scenario_mode: str,
    evidence_packet: str = "",
    resolved_timeline: Optional[List[YearBlock]] = None,
    final_round_summary: Optional[DiscussionSummary] = None,
    agent_positions: Optional[Dict[str, str]] = None,
    allowed_chunk_ids: Optional[List[str]] = None,
    rag_recorder: Optional[Any] = None,
    *,
    evidence_blob: Optional[str] = None,
) -> tuple:
    """Returns (AgentOutput, List[RedTeamFinding])."""
    system = prompt_mod.RED_TEAM_SYSTEM + prompt_mod.SAFETY_TAIL + prompt_mod.JSON_TAIL

    summary_text = (
        json.dumps(final_round_summary.model_dump(), ensure_ascii=False)
        if final_round_summary is not None
        else "<no discussion summary>"
    )

    positions_text = ""
    if agent_positions:
        positions_text = "\n".join(
            "- {k}: {v}".format(k=k, v=v[:300]) for k, v in agent_positions.items()
        )

    packet = evidence_packet or evidence_blob or ""
    timeline_text = format_resolved_timeline(resolved_timeline or [])
    rt_year = resolved_timeline[-1].year if resolved_timeline else 2031
    fixed_suffix = (
        "Return JSON with:\n"
        + prompt_mod.domain_agent_schema_hint(rt_year)
        + CITATION_SCHEMA_APPENDIX
        + "\nPlus a 'findings' array of "
        '{"issue": "...", "severity": "low|medium|high", '
        '"affected_assumption": "..."} objects.'
    )
    max_chars = _config_mod.CONFIG.max_agent_input_chars
    context = budget_prompt_sections(
        [
            ("Seed: " + seed + "\nScenario mode: " + scenario_mode, 1),
            ("Locked year-by-year timeline:\n" + timeline_text, 2),
            ("Evidence packet:\n" + packet, 5),
            ("Final year discussion summary:\n" + summary_text, 4),
            ("Final agent positions:\n" + (positions_text or "<none>"), 3),
        ],
        max_chars - len(fixed_suffix),
    )
    user = context + fixed_suffix

    fallback = _domain_fallback("red_team", 1, 2031)
    fallback["findings"] = [
        {
            "issue": "Linear escalation assumption may be too smooth.",
            "severity": "medium",
            "affected_assumption": "Steady decoupling",
        }
    ]
    data = llm.call_llm_json(
        system_prompt=system,
        user_prompt=user,
        agent_name="red_team",
        round_number=99,
        schema_name="red_team",
        cache_context={"seed": seed, "mode": scenario_mode},
        fallback=fallback,
    )

    data, _warnings = validate_and_apply_citations(
        data, allowed_chunk_ids or [], "red_team", rag_recorder
    )
    output = _to_agent_output("red_team", round_number=99, target_year=0, data=data)
    findings: List[RedTeamFinding] = []
    for raw in data.get("findings") or []:
        if not isinstance(raw, dict):
            continue
        findings.append(
            RedTeamFinding(
                issue=str(raw.get("issue") or "")[:300],
                severity=str(raw.get("severity") or "medium")[:10],
                affected_assumption=str(raw.get("affected_assumption") or "")[:200],
            )
        )
    return output, findings


# --- Orchestrator agents ---------------------------------------------------


def run_orchestrator_summary(
    llm: LLMClient,
    seed: str,
    scenario_mode: str,
    round_number: int,
    target_year: int,
    latest_outputs: Dict[str, AgentOutput],
) -> DiscussionSummary:
    """LLM-backed compaction of a discussion round for one year."""
    from .cost_control import build_discussion_summary

    positions_blob = "\n".join(
        "- {a}: {p}".format(
            a=name,
            p=truncate(out.main_assessment + " | drivers=" + ",".join(out.key_drivers[:3]), 350),
        )
        for name, out in latest_outputs.items()
    )

    system = prompt_mod.ORCHESTRATOR_SUMMARY_SYSTEM + prompt_mod.JSON_TAIL

    fixed_suffix = (
        "Return JSON:\n"
        "{\n"
        '  "round_number": ' + str(round_number) + ",\n"
        '  "target_year": ' + str(target_year) + ",\n"
        '  "areas_of_agreement": [],\n'
        '  "areas_of_disagreement": [],\n'
        '  "emerging_timeline": ["' + str(target_year) + ': ..."],\n'
        '  "key_uncertainties": [],\n'
        '  "agent_positions": {"geo_strategy": "...", "economy_technology": "..."}\n'
        "}\n"
    )
    max_chars = _config_mod.CONFIG.max_agent_input_chars
    context = budget_prompt_sections(
        [
            (
                "Seed: " + seed + "\n"
                "Mode: " + scenario_mode + "\n"
                "Target year: " + str(target_year) + "\n"
                "Round: " + str(round_number),
                1,
            ),
            ("Agent positions:\n" + positions_blob, 3),
        ],
        max_chars - len(fixed_suffix),
    )
    user = context + fixed_suffix

    heuristic = build_discussion_summary(round_number, target_year, latest_outputs)
    data = llm.call_llm_json(
        system_prompt=system,
        user_prompt=user,
        agent_name="orchestrator_summary",
        round_number=round_number,
        schema_name="discussion_summary",
        cache_context={"round": round_number, "year": target_year, "n": len(latest_outputs)},
        fallback=heuristic.model_dump(),
    )

    try:
        return DiscussionSummary(
            round_number=int(data.get("round_number") or round_number),
            target_year=int(data.get("target_year") or target_year),
            areas_of_agreement=_as_str_list(data.get("areas_of_agreement")),
            areas_of_disagreement=_as_str_list(data.get("areas_of_disagreement")),
            emerging_timeline=_as_str_list(data.get("emerging_timeline")),
            key_uncertainties=_as_str_list(data.get("key_uncertainties")),
            agent_positions={
                k: str(v)[:400]
                for k, v in (data.get("agent_positions") or {}).items()
                if isinstance(k, str)
            },
        )
    except Exception:
        return heuristic


def run_orchestrator_year_decision(
    llm: LLMClient,
    seed: str,
    scenario_mode: str,
    target_year: int,
    discussion_rounds: List[DiscussionSummary],
    latest_outputs: Dict[str, AgentOutput],
    resolved_timeline: Optional[List[YearBlock]] = None,
) -> YearBlock:
    """Lock the canonical outcome for one simulation year."""
    summaries_blob = "\n\n".join(
        "Round {r}: {json}".format(
            r=s.round_number,
            json=truncate(json.dumps(s.model_dump(), ensure_ascii=False), 800),
        )
        for s in discussion_rounds
    )
    positions_blob = "\n".join(
        "- {a}: {p}".format(a=name, p=truncate(out.main_assessment, 250))
        for name, out in latest_outputs.items()
    )
    prior = format_resolved_timeline(resolved_timeline or [])

    system = (
        prompt_mod.ORCHESTRATOR_YEAR_DECISION_SYSTEM + prompt_mod.SAFETY_TAIL + prompt_mod.JSON_TAIL
    )
    fixed_suffix = (
        "Return JSON:\n"
        "{\n"
        '  "year": ' + str(target_year) + ",\n"
        '  "headline": "<one sentence for the year>",\n'
        '  "events": [\n'
        "    {\n"
        '      "event": "<short>",\n'
        '      "domain": "strategy|economy|security|ideology|historical|technology",\n'
        '      "probability": 0.5,\n'
        '      "impact": "low|medium|high",\n'
        '      "confidence": "low|medium|high",\n'
        '      "rationale": "<one sentence>"\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )
    max_chars = _config_mod.CONFIG.max_agent_input_chars
    context = budget_prompt_sections(
        [
            (
                "Seed: " + seed + "\n"
                "Mode: " + scenario_mode + "\n"
                "Target year to lock: " + str(target_year),
                1,
            ),
            ("Locked prior years:\n" + prior, 2),
            (
                "Discussion summaries this year:\n" + (summaries_blob or "<none>"),
                4,
            ),
            (
                "Final agent positions this year:\n" + (positions_blob or "<none>"),
                3,
            ),
        ],
        max_chars - len(fixed_suffix),
    )
    user = context + fixed_suffix

    fallback = _year_decision_fallback(target_year, latest_outputs)
    data = llm.call_llm_json(
        system_prompt=system,
        user_prompt=user,
        agent_name="orchestrator_year_decision",
        round_number=target_year,
        schema_name="year_decision",
        cache_context={"seed": seed, "mode": scenario_mode, "year": target_year},
        fallback=fallback.model_dump(),
    )
    return _year_block_from_dict(data, target_year, fallback)


def run_orchestrator_final_synthesis_raw(
    llm: LLMClient,
    seed: str,
    scenario_mode: str,
    evidence: EvidenceSummary,
    resolved_timeline: List[YearBlock],
    last_summary: Optional[DiscussionSummary],
    domain_outputs: Dict[str, AgentOutput],
    red_team: AgentOutput,
    red_team_findings: List[RedTeamFinding],
    final_evidence_packet: Optional[FinalEvidencePacket] = None,
    regeneration: bool = False,
) -> Dict[str, Any]:
    """Call the final orchestrator LLM and return raw JSON (unvalidated).

    Validation/repair is applied in `final_output_validation.resolve_orchestrator_synthesis`
    from the graph node only.
    """
    system = (
        prompt_mod.ORCHESTRATOR_FINAL_SYSTEM + prompt_mod.SAFETY_TAIL + prompt_mod.JSON_TAIL
    )

    positions = "\n".join(
        "- {n}: {p}".format(n=k, p=truncate(v.main_assessment, 200))
        for k, v in domain_outputs.items()
    )
    findings = "\n".join(
        "- [{s}] {i}".format(s=f.severity, i=f.issue) for f in red_team_findings[:6]
    )
    last_summary_txt = (
        json.dumps(last_summary.model_dump(), ensure_ascii=False)
        if last_summary is not None
        else "<none>"
    )

    packet_txt = ""
    if final_evidence_packet and final_evidence_packet.text:
        packet_txt = final_evidence_packet.text
    elif final_evidence_packet and final_evidence_packet.items:
        packet_txt = "\n".join(final_evidence_packet.items)

    timeline_txt = format_resolved_timeline(resolved_timeline)

    fixed_suffix = (
        "Return JSON:\n"
        "{\n"
        '  "scenario_title": "<short>",\n'
        '  "scenario_summary": "<3-5 sentences>",\n'
        '  "event_status": "observed|hypothetical|mixed",\n'
        '  "key_assumptions": ["..."],\n'
        '  "main_disagreements": ["..."],\n'
        '  "image_prompt": "<editorial illustration prompt>"\n'
        "}\n"
    )
    max_chars = _config_mod.CONFIG.max_agent_input_chars
    context = budget_prompt_sections(
        [
            (
                "Seed: " + seed + "\n"
                "Mode: " + scenario_mode + "\n"
                "Evidence note: " + (evidence.note or "") + "\n"
                "Compact evidence: " + (evidence.compact_summary or ""),
                1,
            ),
            ("Locked timeline (do not rewrite):\n" + timeline_txt, 2),
            (
                "Final evidence packet (curated, cite-aware):\n"
                + (packet_txt or "<none>"),
                5,
            ),
            ("Final year discussion summary:\n" + last_summary_txt, 4),
            ("Last-year agent assessments:\n" + positions, 3),
            ("Red-team findings:\n" + (findings or "<none>"), 3),
        ],
        max_chars - len(fixed_suffix),
    )
    user = context + fixed_suffix

    fallback = {
        "scenario_title": "One Plausible USA-China Rivalry Path (2026-2031)",
        "scenario_summary": (
            "A synthesized, non-predictive scenario built from multi-agent "
            "analysis of " + truncate(seed, 200) + "."
        ),
        "event_status": "hypothetical",
        "key_assumptions": ["No major hot war between great powers."],
        "main_disagreements": [d for d in (last_summary.areas_of_disagreement if last_summary else [])][:5],
        "image_prompt": "",
    }
    cache_context = {"seed": seed, "mode": scenario_mode}
    if regeneration:
        cache_context["regeneration"] = True

    data = llm.call_llm_json(
        system_prompt=system,
        user_prompt=user,
        agent_name="orchestrator_final",
        round_number=999,
        schema_name="final_synthesis",
        cache_context=cache_context,
        fallback=fallback,
    )
    return data if isinstance(data, dict) else fallback


def run_orchestrator_final_synthesis(
    llm: LLMClient,
    seed: str,
    scenario_mode: str,
    evidence: EvidenceSummary,
    resolved_timeline: List[YearBlock],
    last_summary: Optional[DiscussionSummary],
    domain_outputs: Dict[str, AgentOutput],
    red_team: AgentOutput,
    red_team_findings: List[RedTeamFinding],
) -> Dict[str, Any]:
    """Backward-compatible wrapper returning raw orchestrator JSON."""
    return run_orchestrator_final_synthesis_raw(
        llm,
        seed,
        scenario_mode,
        evidence,
        resolved_timeline,
        last_summary,
        domain_outputs,
        red_team,
        red_team_findings,
    )


def classify_event_status(seed: str) -> str:
    """Cheap, deterministic classifier used during state init.

    The Orchestrator's LLM synthesis may override this later. We use it
    so the API can echo back an event_status even if synthesis fails.
    """
    s = (seed or "").lower()
    future_markers = (
        "will", "would", "if ", "suppose", "hypothetical", "imagine",
        "what if", "scenario where", "unexpectedly",
    )
    past_markers = ("happened", "yesterday", "last year", "in 2024", "in 2023")
    has_future = any(m in s for m in future_markers)
    has_past = any(m in s for m in past_markers)
    if has_future and has_past:
        return "mixed"
    if has_future:
        return "hypothetical"
    return "hypothetical"


# --- Timeline assembly -----------------------------------------------------


def build_final_timeline(resolved_timeline: List[YearBlock]) -> List[YearBlock]:
    """Return the orchestrator-locked timeline (already built year-by-year)."""
    if resolved_timeline:
        return list(resolved_timeline)
    return [YearBlock(year=y) for y in SIMULATION_YEARS]


# --- Internals -------------------------------------------------------------


def _as_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v)[:400] for v in value if v is not None]
    return []


def _to_agent_output(
    agent_name: str,
    round_number: int,
    target_year: int,
    data: Dict[str, Any],
) -> AgentOutput:
    contribs: List[AgentTimelineContribution] = []
    for raw in data.get("timeline_contributions") or []:
        if not isinstance(raw, dict):
            continue
        try:
            year = int(raw.get("year") or target_year or 2026)
            if target_year and year != target_year:
                continue
            contribs.append(
                AgentTimelineContribution(
                    year=year,
                    event=str(raw.get("event") or "")[:300],
                    probability=float(raw.get("probability") or 0.5),
                    impact=str(raw.get("impact") or "medium")[:10],
                    confidence=str(raw.get("confidence") or "medium")[:10],
                    rationale=str(raw.get("rationale") or "")[:400],
                )
            )
        except Exception:
            continue
    out = AgentOutput(
        agent_name=agent_name,
        round_number=round_number,
        target_year=target_year,
        main_assessment=str(data.get("main_assessment") or "")[:1500],
        key_drivers=_as_str_list(data.get("key_drivers")),
        timeline_contributions=contribs,
        risks=_as_str_list(data.get("risks")),
        uncertainties=_as_str_list(data.get("uncertainties")),
        agreements=_as_str_list(data.get("agreements")),
        disagreements=_as_str_list(data.get("disagreements")),
        position_changed_from_previous_round=bool(
            data.get("position_changed_from_previous_round")
        ),
    )
    return apply_citations_to_output(out, data)


def _year_block_from_dict(
    data: Dict[str, Any],
    target_year: int,
    fallback: YearBlock,
) -> YearBlock:
    try:
        events: List[TimelineEvent] = []
        for raw in data.get("events") or []:
            if not isinstance(raw, dict):
                continue
            events.append(
                TimelineEvent(
                    event=str(raw.get("event") or "")[:300],
                    domain=str(raw.get("domain") or "strategy")[:20],
                    probability=float(raw.get("probability") or 0.5),
                    impact=str(raw.get("impact") or "medium")[:10],
                    confidence=str(raw.get("confidence") or "medium")[:10],
                    rationale=str(raw.get("rationale") or "")[:400],
                )
            )
        headline = str(data.get("headline") or "")[:200]
        if not headline and events:
            headline = truncate(events[0].event, 140)
        if not headline:
            return fallback
        return YearBlock(
            year=int(data.get("year") or target_year),
            headline=headline,
            events=events or fallback.events,
        )
    except Exception:
        return fallback


def _year_decision_fallback(
    target_year: int,
    latest_outputs: Dict[str, AgentOutput],
) -> YearBlock:
    domain_for_agent = {
        "geo_strategy": "strategy",
        "economy_technology": "economy",
        "domestic_ideology": "ideology",
        "security_taiwan": "security",
        "historical_analogy": "historical",
    }
    events: List[TimelineEvent] = []
    for agent_name, output in latest_outputs.items():
        domain = domain_for_agent.get(agent_name, "strategy")
        for tc in output.timeline_contributions:
            if tc.year and tc.year != target_year:
                continue
            if not tc.event:
                continue
            events.append(
                TimelineEvent(
                    event=tc.event,
                    domain=domain,
                    probability=tc.probability,
                    impact=tc.impact,
                    confidence=tc.confidence,
                    rationale=tc.rationale,
                )
            )
    headline = truncate(events[0].event, 140) if events else (
        "Scenario developments continue in " + str(target_year)
    )
    return YearBlock(year=target_year, headline=headline, events=events[:6])


def _domain_fallback(agent_name: str, round_number: int, target_year: int) -> Dict[str, Any]:
    return {
        "agent_name": agent_name,
        "round_number": round_number,
        "main_assessment": "[fallback] structured response unavailable.",
        "key_drivers": [],
        "timeline_contributions": [
            {
                "year": target_year,
                "event": "[fallback] no input from " + agent_name,
                "probability": 0.3,
                "impact": "low",
                "confidence": "low",
                "rationale": "fallback",
            }
        ],
        "risks": [],
        "uncertainties": [],
        "agreements": [],
        "disagreements": [],
        "position_changed_from_previous_round": False,
        "sources_used": [],
        "grounding_notes": [],
        "rag_influence": "not_used",
        "rag_influence_explanation": "",
    }
