"""Run domain agents in parallel via LangChain RunnableParallel."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from . import agents as agent_mod
from . import config as _config_mod
from .cost_control import compact_agent_position
from .llm import LLMClient
from .rag import RagMetricsRecorder, retrieve_for_agent
from .rag_citations import build_agent_evidence_packet, register_chunks
from .schemas import AgentOutput, DiscussionSummary, DOMAIN_AGENTS, EvidenceChunk, EvidenceLanes, ScenarioState


@dataclass
class _AgentJob:
    agent_name: str
    evidence_packet: str
    allowed_chunk_ids: List[str]
    previous_self_position: Optional[str]


def _run_agent_job(
    job: _AgentJob,
    llm: LLMClient,
    seed: str,
    scenario_mode: str,
    target_year: int,
    resolved_timeline,
    round_number: int,
    previous_summary: Optional[DiscussionSummary],
    recorder: RagMetricsRecorder,
    metrics_lock: threading.Lock,
) -> Tuple[str, AgentOutput]:
    """Execute one domain agent (thread-safe metrics via lock)."""
    with metrics_lock:
        output = agent_mod.run_domain_agent(
            llm=llm,
            agent_name=job.agent_name,
            seed=seed,
            scenario_mode=scenario_mode,
            target_year=target_year,
            resolved_timeline=resolved_timeline,
            evidence_packet=job.evidence_packet,
            round_number=round_number,
            previous_summary=previous_summary,
            previous_self_position=job.previous_self_position,
            allowed_chunk_ids=job.allowed_chunk_ids,
            rag_recorder=recorder,
        )
    return job.agent_name, output


def _prepare_jobs(
    state: ScenarioState,
    round_number: int,
    recorder: RagMetricsRecorder,
    previous_summary: Optional[DiscussionSummary],
    disagreement_chunks: List[EvidenceChunk],
) -> List[_AgentJob]:
    jobs: List[_AgentJob] = []
    for agent_name in DOMAIN_AGENTS:
        prev_self = None
        history = state.agent_outputs.get(agent_name) or []
        if history:
            prev_self = compact_agent_position(history[-1])

        agent_chunks: List[EvidenceChunk] = []
        if round_number == 1 and _config_mod.CONFIG.use_rag:
            agent_chunks = retrieve_for_agent(
                state.seed,
                state.scenario_mode,
                agent_name,
                target_year=state.current_year,
                round_number=1,
                recorder=recorder,
            )
            register_chunks(state.chunks_used_registry, agent_chunks)

        packet, allowed_ids = build_agent_evidence_packet(
            agent_name,
            state.evidence_lanes,
            agent_chunks,
            disagreement_chunks=disagreement_chunks,
            round_number=round_number,
        )
        register_chunks(state.chunks_used_registry, agent_chunks)
        register_chunks(state.chunks_used_registry, disagreement_chunks)

        jobs.append(
            _AgentJob(
                agent_name=agent_name,
                evidence_packet=packet,
                allowed_chunk_ids=allowed_ids,
                previous_self_position=prev_self,
            )
        )
    return jobs


def run_domain_agents_parallel(
    state: ScenarioState,
    llm: LLMClient,
    round_number: int,
    recorder: RagMetricsRecorder,
    previous_summary: Optional[DiscussionSummary],
    disagreement_chunks: List[EvidenceChunk],
) -> Dict[str, AgentOutput]:
    """Run all five domain agents concurrently using RunnableParallel."""
    jobs = _prepare_jobs(
        state, round_number, recorder, previous_summary, disagreement_chunks
    )
    metrics_lock = threading.Lock()

    try:
        from langchain_core.runnables import RunnableLambda, RunnableParallel
    except ImportError:
        return _run_domain_agents_sequential_fallback(
            state, llm, round_number, recorder, previous_summary, disagreement_chunks
        )

    branches: Dict[str, Any] = {}
    for job in jobs:
        branches[job.agent_name] = RunnableLambda(
            lambda _input, j=job: _run_agent_job(
                j,
                llm,
                state.seed,
                state.scenario_mode,
                state.current_year,
                state.resolved_timeline,
                round_number,
                previous_summary,
                recorder,
                metrics_lock,
            )
        )

    parallel = RunnableParallel(**branches)
    results: Dict[str, Tuple[str, AgentOutput]] = parallel.invoke({})

    latest: Dict[str, AgentOutput] = {}
    for agent_name in DOMAIN_AGENTS:
        name, output = results[agent_name]
        assert name == agent_name
        state.agent_outputs.setdefault(agent_name, []).append(output)
        latest[agent_name] = output
    return latest


def _run_domain_agents_sequential_fallback(
    state: ScenarioState,
    llm: LLMClient,
    round_number: int,
    recorder: RagMetricsRecorder,
    previous_summary: Optional[DiscussionSummary],
    disagreement_chunks: List[EvidenceChunk],
) -> Dict[str, AgentOutput]:
    jobs = _prepare_jobs(
        state, round_number, recorder, previous_summary, disagreement_chunks
    )
    latest: Dict[str, AgentOutput] = {}
    for job in jobs:
        _name, output = _run_agent_job(
            job,
            llm,
            state.seed,
            state.scenario_mode,
            state.current_year,
            state.resolved_timeline,
            round_number,
            previous_summary,
            recorder,
            threading.Lock(),
        )
        state.agent_outputs.setdefault(job.agent_name, []).append(output)
        latest[job.agent_name] = output
    return latest
