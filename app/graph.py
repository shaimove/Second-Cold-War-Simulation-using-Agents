"""LangGraph workflow that wires every agent together.

Flow:
    START -> orchestrator_initialize -> evidence_rag_agent
          -> discussion_round_1 -> orchestrator_summarize_round_1
          -> disagreement_retrieval (round 2 prep)
          -> discussion_round_2 -> orchestrator_summarize_round_2
          -> discussion_round_3 -> orchestrator_summarize_round_3
          -> red_team_agent -> orchestrator_synthesis
          -> orchestrator_image_generation -> save_run -> END
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from . import agents as agent_mod
from . import config as _config_mod
from . import db
from .cost_control import compact_agent_position, should_stop_early
from .final_output_validation import resolve_orchestrator_synthesis
from .image_generation import build_image_prompt, generate_image
from .llm import LLMClient
from .rag import (
    RagMetricsRecorder,
    build_evidence_lanes,
    build_final_evidence_packet,
    clear_retrieval_cache,
    retrieve_baseline,
    retrieve_for_agent,
    retrieve_for_disagreement,
    retrieve_for_red_team,
)
from .rag_citations import build_agent_evidence_packet, register_chunks
from .schemas import (
    AgentOutput,
    DiscussionSummary,
    DOMAIN_AGENTS,
    FinalScenario,
    ImageResult,
    RunMetrics,
    ScenarioState,
)
from .utils import new_run_id, truncate


def _rag_recorder(state: ScenarioState) -> RagMetricsRecorder:
    return RagMetricsRecorder(metrics=state.run_metrics)


def orchestrator_initialize(state: ScenarioState, _llm: LLMClient) -> ScenarioState:
    if not state.event_status:
        state.event_status = agent_mod.classify_event_status(state.seed)
    if not state.scenario_title:
        state.scenario_title = "USA-China Scenario: " + truncate(state.seed, 80)
    for name in DOMAIN_AGENTS:
        state.agent_outputs.setdefault(name, [])
    clear_retrieval_cache()
    return state


def evidence_rag_node(state: ScenarioState, llm: LLMClient) -> ScenarioState:
    recorder = _rag_recorder(state)
    baseline: List = []
    if _config_mod.CONFIG.use_rag:
        baseline = retrieve_baseline(state.seed, state.scenario_mode, recorder=recorder)
    state.baseline_chunks = baseline
    register_chunks(state.chunks_used_registry, baseline)

    summary, n_chunks, _cache_hit = agent_mod.run_evidence_agent(
        llm, state.seed, state.scenario_mode, baseline_chunks=baseline
    )
    state.evidence_summary = summary
    state.evidence_lanes = build_evidence_lanes(baseline, summary)
    state.run_metrics.retrieved_docs = n_chunks
    return state


def _run_discussion_round(
    state: ScenarioState, llm: LLMClient, round_number: int
) -> ScenarioState:
    if getattr(state, "_early_stopped", False):
        return state

    recorder = _rag_recorder(state)
    prev_summary: Optional[DiscussionSummary] = (
        state.discussion_rounds[-1] if state.discussion_rounds else None
    )
    disagreement_chunks = (
        state.disagreement_chunks if round_number >= 2 else []
    )

    latest_outputs: Dict[str, AgentOutput] = {}
    for agent_name in DOMAIN_AGENTS:
        prev_self = None
        history = state.agent_outputs.get(agent_name) or []
        if history:
            prev_self = compact_agent_position(history[-1])

        agent_chunks: List = []
        if round_number == 1 and _config_mod.CONFIG.use_rag:
            agent_chunks = retrieve_for_agent(
                state.seed,
                state.scenario_mode,
                agent_name,
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

        output = agent_mod.run_domain_agent(
            llm=llm,
            agent_name=agent_name,
            seed=state.seed,
            scenario_mode=state.scenario_mode,
            evidence_packet=packet,
            round_number=round_number,
            previous_summary=prev_summary,
            previous_self_position=prev_self,
            allowed_chunk_ids=allowed_ids,
            rag_recorder=recorder,
        )
        state.agent_outputs.setdefault(agent_name, []).append(output)
        latest_outputs[agent_name] = output

    state.run_metrics.discussion_rounds_completed = round_number
    state._latest_round_outputs = latest_outputs  # type: ignore[attr-defined]
    return state


def discussion_round_1(state: ScenarioState, llm: LLMClient) -> ScenarioState:
    return _run_discussion_round(state, llm, 1)


def disagreement_retrieval_node(state: ScenarioState, _llm: LLMClient) -> ScenarioState:
    if not state.discussion_rounds:
        return state
    recorder = _rag_recorder(state)
    chunks = retrieve_for_disagreement(
        state.seed,
        state.scenario_mode,
        state.discussion_rounds[-1],
        recorder=recorder,
    )
    state.disagreement_chunks = chunks
    register_chunks(state.chunks_used_registry, chunks)
    return state


def discussion_round_2(state: ScenarioState, llm: LLMClient) -> ScenarioState:
    if getattr(state, "_early_stopped", False):
        return state
    return _run_discussion_round(state, llm, 2)


def discussion_round_3(state: ScenarioState, llm: LLMClient) -> ScenarioState:
    if getattr(state, "_early_stopped", False):
        return state
    if _config_mod.CONFIG.max_agent_discussion_rounds < 3:
        return state
    return _run_discussion_round(state, llm, 3)


def _summarize_round(state: ScenarioState, llm: LLMClient, round_number: int) -> ScenarioState:
    latest = getattr(state, "_latest_round_outputs", None)
    if not latest:
        return state
    summary = agent_mod.run_orchestrator_summary(
        llm,
        seed=state.seed,
        scenario_mode=state.scenario_mode,
        round_number=round_number,
        latest_outputs=latest,
    )
    state.discussion_rounds.append(summary)

    if round_number >= 2 and should_stop_early(round_number, summary):
        state._early_stopped = True  # type: ignore[attr-defined]
    return state


def orchestrator_summarize_round_1(state: ScenarioState, llm: LLMClient) -> ScenarioState:
    return _summarize_round(state, llm, 1)


def orchestrator_summarize_round_2(state: ScenarioState, llm: LLMClient) -> ScenarioState:
    if getattr(state, "_early_stopped", False) and state.run_metrics.discussion_rounds_completed < 2:
        return state
    return _summarize_round(state, llm, 2)


def orchestrator_summarize_round_3(state: ScenarioState, llm: LLMClient) -> ScenarioState:
    if state.run_metrics.discussion_rounds_completed < 3:
        return state
    return _summarize_round(state, llm, 3)


def red_team_node(state: ScenarioState, llm: LLMClient) -> ScenarioState:
    last_summary = state.discussion_rounds[-1] if state.discussion_rounds else None
    recorder = _rag_recorder(state)
    red_chunks = retrieve_for_red_team(
        state.seed,
        state.scenario_mode,
        last_summary,
        recorder=recorder,
    )
    state.red_team_chunks = red_chunks
    register_chunks(state.chunks_used_registry, red_chunks)

    positions: Dict[str, str] = {}
    if last_summary and last_summary.agent_positions:
        positions = last_summary.agent_positions
    else:
        for name in DOMAIN_AGENTS:
            outs = state.agent_outputs.get(name) or []
            if outs:
                positions[name] = truncate(outs[-1].main_assessment, 300)

    packet, allowed_ids = build_agent_evidence_packet(
        "red_team",
        state.evidence_lanes,
        red_chunks,
        round_number=99,
    )
    packet = packet + "\n\nRed-team critique chunks included above."

    output, findings = agent_mod.run_red_team_agent(
        llm,
        seed=state.seed,
        scenario_mode=state.scenario_mode,
        evidence_packet=packet,
        final_round_summary=last_summary,
        agent_positions=positions,
        allowed_chunk_ids=allowed_ids,
        rag_recorder=recorder,
    )
    state.agent_outputs.setdefault("red_team", []).append(output)
    state.red_team_findings = findings
    return state


def orchestrator_synthesis(state: ScenarioState, llm: LLMClient) -> ScenarioState:
    last_summary = state.discussion_rounds[-1] if state.discussion_rounds else None
    last_per_agent: Dict[str, AgentOutput] = {
        name: outs[-1]
        for name, outs in state.agent_outputs.items()
        if outs and name in DOMAIN_AGENTS
    }
    red_team_history = state.agent_outputs.get("red_team") or []
    red_team_last = red_team_history[-1] if red_team_history else AgentOutput(
        agent_name="red_team", main_assessment=""
    )

    state.final_timeline = agent_mod.build_final_timeline(last_per_agent)

    state.final_evidence_packet = build_final_evidence_packet(
        baseline_chunks=state.baseline_chunks,
        disagreement_chunks=state.disagreement_chunks,
        red_team_chunks=state.red_team_chunks,
        lanes=state.evidence_lanes,
        agent_outputs=state.agent_outputs,
    )

    raw_synthesis = agent_mod.run_orchestrator_final_synthesis_raw(
        llm,
        seed=state.seed,
        scenario_mode=state.scenario_mode,
        evidence=state.evidence_summary,
        last_summary=last_summary,
        domain_outputs=last_per_agent,
        red_team=red_team_last,
        red_team_findings=state.red_team_findings,
        final_evidence_packet=state.final_evidence_packet,
    )

    def _regenerate_synthesis():
        return agent_mod.run_orchestrator_final_synthesis_raw(
            llm,
            seed=state.seed,
            scenario_mode=state.scenario_mode,
            evidence=state.evidence_summary,
            last_summary=last_summary,
            domain_outputs=last_per_agent,
            red_team=red_team_last,
            red_team_findings=state.red_team_findings,
            final_evidence_packet=state.final_evidence_packet,
            regeneration=True,
        )

    synthesis, _passed, image_disabled = resolve_orchestrator_synthesis(
        llm=llm,
        raw_output=raw_synthesis,
        seed=state.seed,
        scenario_mode=state.scenario_mode,
        discussion_summary=last_summary,
        regenerate_fn=_regenerate_synthesis,
        partial_timeline=state.final_timeline,
        metrics=state.run_metrics,
    )

    state.scenario_title = synthesis.get("scenario_title") or state.scenario_title
    state.scenario_summary = synthesis.get("scenario_summary") or ""
    state.event_status = synthesis.get("event_status") or state.event_status
    state.disagreements = synthesis.get("main_disagreements") or []

    if image_disabled or synthesis.get("image_generation_disabled"):
        state.image_prompt = ""
        state.image_result = ImageResult(enabled=False, generated=False)
        state.errors.append(
            synthesis.get("error")
            or state.run_metrics.synthesis_error_message
            or "synthesis_fallback"
        )
    else:
        state.image_prompt = synthesis.get("image_prompt") or build_image_prompt(
            state.scenario_title, state.scenario_summary
        )

    return state


def orchestrator_image_generation(state: ScenarioState, _llm: LLMClient) -> ScenarioState:
    if not state.image_prompt:
        state.image_prompt = build_image_prompt(
            state.scenario_title, state.scenario_summary
        )
    result = generate_image(state.run_id, state.image_prompt)
    state.image_result = result
    return state


def save_run_node(state: ScenarioState, _llm: LLMClient) -> ScenarioState:
    final = build_final_scenario(state)
    try:
        db.save_scenario_run(
            run_id=state.run_id,
            seed=state.seed,
            scenario_mode=state.scenario_mode,
            scenario_title=state.scenario_title,
            full_json=final.model_dump(),
        )
    except Exception as e:
        state.errors.append("save_failed: " + str(e))
    return state


def build_final_scenario(state: ScenarioState) -> FinalScenario:
    agent_summaries: Dict[str, str] = {}
    for name in DOMAIN_AGENTS:
        outs = state.agent_outputs.get(name) or []
        if outs:
            agent_summaries[name] = truncate(outs[-1].main_assessment, 400)
    red_outs = state.agent_outputs.get("red_team") or []
    if red_outs:
        agent_summaries["red_team"] = truncate(red_outs[-1].main_assessment, 400)

    red_warnings: List[str] = [f.issue for f in state.red_team_findings]
    last_per_agent: Dict[str, AgentOutput] = {
        name: outs[-1]
        for name, outs in state.agent_outputs.items()
        if outs and name in DOMAIN_AGENTS
    }
    key_assumptions: List[str] = []
    for out in last_per_agent.values():
        key_assumptions.extend(out.agreements[:2])
    key_assumptions = list(dict.fromkeys(a for a in key_assumptions if a))[:8]

    return FinalScenario(
        run_id=state.run_id,
        scenario_title=state.scenario_title,
        scenario_summary=state.scenario_summary,
        seed=state.seed,
        scenario_mode=state.scenario_mode,
        event_status=state.event_status,
        timeline=state.final_timeline,
        key_assumptions=key_assumptions,
        main_disagreements=state.disagreements,
        red_team_warnings=red_warnings,
        agent_summaries=agent_summaries,
        discussion_summary=state.discussion_rounds,
        image_prompt=state.image_prompt,
        image=state.image_result,
        run_metrics=state.run_metrics,
    )


NODES: List[tuple] = [
    ("orchestrator_initialize", orchestrator_initialize),
    ("evidence_rag_agent", evidence_rag_node),
    ("discussion_round_1", discussion_round_1),
    ("orchestrator_summarize_round_1", orchestrator_summarize_round_1),
    ("disagreement_retrieval", disagreement_retrieval_node),
    ("discussion_round_2", discussion_round_2),
    ("orchestrator_summarize_round_2", orchestrator_summarize_round_2),
    ("discussion_round_3", discussion_round_3),
    ("orchestrator_summarize_round_3", orchestrator_summarize_round_3),
    ("red_team_agent", red_team_node),
    ("orchestrator_synthesis", orchestrator_synthesis),
    ("orchestrator_image_generation", orchestrator_image_generation),
    ("save_run", save_run_node),
]


def build_graph():
    try:
        from langgraph.graph import StateGraph, END  # type: ignore
    except Exception:
        return None

    graph = StateGraph(dict)

    def make_wrapper(fn: Callable, name: str):
        def _w(state_dict: Dict[str, Any]) -> Dict[str, Any]:
            llm: LLMClient = state_dict["_llm"]
            ss = ScenarioState(
                **{k: v for k, v in state_dict.items() if not k.startswith("_")}
            )
            for k, v in state_dict.items():
                if k.startswith("_") and k != "_llm":
                    setattr(ss, k, v)
            ss = fn(ss, llm)
            out = ss.model_dump()
            out["_llm"] = llm
            for k in ("_early_stopped", "_latest_round_outputs"):
                if hasattr(ss, k):
                    out[k] = getattr(ss, k)
            return out

        _w.__name__ = name
        return _w

    prev = None
    for name, fn in NODES:
        graph.add_node(name, make_wrapper(fn, name))
        if prev is None:
            graph.set_entry_point(name)
        else:
            graph.add_edge(prev, name)
        prev = name
    graph.add_edge(prev, END)
    return graph.compile()


def run_graph(
    seed: str,
    scenario_mode: str,
    llm: Optional[LLMClient] = None,
) -> FinalScenario:
    llm = llm or LLMClient()
    state = ScenarioState(run_id=new_run_id(), seed=seed, scenario_mode=scenario_mode)
    state.run_metrics = RunMetrics()
    start = time.time()

    compiled = build_graph()
    if compiled is not None:
        try:
            payload = state.model_dump()
            payload["_llm"] = llm
            result = compiled.invoke(payload)
            state = ScenarioState(
                **{k: v for k, v in result.items() if not k.startswith("_")}
            )
        except Exception as e:
            state.errors.append("langgraph_failed: " + str(e))
            state = _run_sequential(state, llm)
    else:
        state = _run_sequential(state, llm)

    state.run_metrics.elapsed_seconds = round(time.time() - start, 3)
    state.run_metrics.llm_calls = llm.metrics.llm_calls
    state.run_metrics.cache_hits = max(
        state.run_metrics.cache_hits, llm.metrics.cache_hits
    )
    state.run_metrics.agents_used = list(llm.metrics.agents_used)
    state.run_metrics.estimated_input_tokens = llm.metrics.estimated_input_tokens
    state.run_metrics.estimated_output_tokens = llm.metrics.estimated_output_tokens

    final = build_final_scenario(state)
    try:
        db.save_scenario_run(
            run_id=state.run_id,
            seed=state.seed,
            scenario_mode=state.scenario_mode,
            scenario_title=state.scenario_title,
            full_json=final.model_dump(),
        )
    except Exception as e:
        state.errors.append("save_failed_final: " + str(e))
    return final


def _run_sequential(state: ScenarioState, llm: LLMClient) -> ScenarioState:
    for _name, fn in NODES:
        try:
            state = fn(state, llm)
        except Exception as e:
            state.errors.append(_name + "_failed: " + str(e))
    return state
