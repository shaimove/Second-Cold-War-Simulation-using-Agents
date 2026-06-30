"""Per-year discussion cycles (up to 3 iterations, then year lock-in)."""
from __future__ import annotations

from typing import Dict, Optional

from . import agents as agent_mod
from . import config as _config_mod
from .cost_control import should_stop_early
from .llm import LLMClient
from .parallel_agents import (
    _run_domain_agents_sequential_fallback as _run_domain_agents_sequential,
    run_domain_agents_parallel,
)
from .rag import RagMetricsRecorder, retrieve_for_disagreement
from .rag_citations import register_chunks
from .monitor.year_judge import evaluate_locked_year
from .schemas import (
    AgentOutput,
    DiscussionSummary,
    DOMAIN_AGENTS,
    ScenarioState,
    YearSimulationRecord,
)


def _reset_year_discussion_state(state: ScenarioState, year: int) -> None:
    state.current_year = year
    state.discussion_rounds = []
    state.disagreement_chunks = []
    state._early_stopped = False  # type: ignore[attr-defined]
    for name in DOMAIN_AGENTS:
        state.agent_outputs[name] = []


def _run_discussion_round(
    state: ScenarioState,
    llm: LLMClient,
    round_number: int,
    recorder: RagMetricsRecorder,
) -> ScenarioState:
    if getattr(state, "_early_stopped", False):
        return state

    prev_summary: Optional[DiscussionSummary] = (
        state.discussion_rounds[-1] if state.discussion_rounds else None
    )
    disagreement_chunks = state.disagreement_chunks if round_number >= 2 else []

    if _config_mod.CONFIG.parallel_domain_agents:
        latest_outputs = run_domain_agents_parallel(
            state,
            llm,
            round_number,
            recorder,
            prev_summary,
            disagreement_chunks,
        )
    else:
        latest_outputs = _run_domain_agents_sequential(
            state, llm, round_number, recorder, prev_summary, disagreement_chunks
        )

    state.run_metrics.discussion_rounds_completed = round_number
    state._latest_round_outputs = latest_outputs  # type: ignore[attr-defined]
    return state


def _summarize_round(
    state: ScenarioState,
    llm: LLMClient,
    round_number: int,
    year: int,
) -> ScenarioState:
    latest: Optional[Dict[str, AgentOutput]] = getattr(
        state, "_latest_round_outputs", None
    )
    if not latest:
        return state

    summary = agent_mod.run_orchestrator_summary(
        llm,
        seed=state.seed,
        scenario_mode=state.scenario_mode,
        round_number=round_number,
        target_year=year,
        latest_outputs=latest,
    )
    state.discussion_rounds.append(summary)

    if round_number >= 2 and should_stop_early(round_number, summary):
        state._early_stopped = True  # type: ignore[attr-defined]
    return state


def _disagreement_retrieval(
    state: ScenarioState,
    year: int,
    recorder: RagMetricsRecorder,
) -> ScenarioState:
    if not state.discussion_rounds:
        return state
    chunks = retrieve_for_disagreement(
        state.seed,
        state.scenario_mode,
        state.discussion_rounds[-1],
        target_year=year,
        recorder=recorder,
    )
    state.disagreement_chunks = chunks
    register_chunks(state.chunks_used_registry, chunks)
    return state


def run_year_cycle(
    state: ScenarioState,
    llm: LLMClient,
    year: int,
    recorder: RagMetricsRecorder,
) -> ScenarioState:
    """Run up to 3 discussion iterations for one year, then lock the year."""
    _reset_year_discussion_state(state, year)
    max_rounds = _config_mod.CONFIG.max_agent_discussion_rounds

    state = _run_discussion_round(state, llm, 1, recorder)
    state = _summarize_round(state, llm, 1, year)

    state = _disagreement_retrieval(state, year, recorder)

    if not getattr(state, "_early_stopped", False):
        state = _run_discussion_round(state, llm, 2, recorder)
        state = _summarize_round(state, llm, 2, year)

    if (
        not getattr(state, "_early_stopped", False)
        and max_rounds >= 3
        and state.run_metrics.discussion_rounds_completed >= 2
    ):
        state = _run_discussion_round(state, llm, 3, recorder)
        state = _summarize_round(state, llm, 3, year)

    latest: Dict[str, AgentOutput] = getattr(state, "_latest_round_outputs", {}) or {}
    year_block = agent_mod.run_orchestrator_year_decision(
        llm,
        seed=state.seed,
        scenario_mode=state.scenario_mode,
        target_year=year,
        discussion_rounds=state.discussion_rounds,
        latest_outputs=latest,
        resolved_timeline=state.resolved_timeline,
    )

    year_gates, year_judge = evaluate_locked_year(
        year,
        state.seed,
        state.scenario_mode,
        year_block,
        state.discussion_rounds,
        latest,
        prior_timeline=list(state.resolved_timeline),
        llm=llm,
    )
    if year_judge:
        state.run_metrics.year_judges_run += 1
        if year_judge.pass_quality_bar:
            state.run_metrics.year_judges_passed += 1
    if year_gates.warnings:
        for w in year_gates.warnings:
            state.run_metrics.citation_warnings.append("year_%d: %s" % (year, w))
    if year_judge and not year_judge.pass_quality_bar:
        state.run_metrics.citation_warnings.append(
            "year_%d judge: %s" % (year, year_judge.one_line_verdict or "below quality bar")
        )

    state.resolved_timeline.append(year_block)
    state.final_timeline = list(state.resolved_timeline)
    state.year_records.append(
        YearSimulationRecord(
            year=year,
            discussion_rounds=list(state.discussion_rounds),
            resolved=year_block,
            year_gates=year_gates,
            year_judge=year_judge,
        )
    )
    state.run_metrics.years_completed = len(state.resolved_timeline)
    return state
