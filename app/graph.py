"""LangGraph workflow that wires every agent together.

Flow:
    START -> orchestrator_initialize -> evidence_rag_agent
          -> year_2026_cycle -> ... -> year_2031_cycle
          -> timeline_quality_check
          -> red_team_agent -> orchestrator_synthesis
          -> orchestrator_image_generation -> save_run -> END

Each year cycle runs up to 3 discussion iterations, then locks that year.
After all six years are locked, timeline QA runs before red team.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from . import agents as agent_mod
from . import config as _config_mod
from . import db
from .final_output_validation import resolve_orchestrator_synthesis
from .image_generation import build_image_prompt, generate_image
from .llm import LLMClient
from .rag import (
    RagMetricsRecorder,
    build_evidence_lanes,
    build_final_evidence_packet,
    clear_retrieval_cache,
    retrieve_baseline,
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
from .checkpoints import (
    get_active_llm,
    get_checkpointer,
    get_checkpoint_status,
    graph_run_config,
    reset_active_llm,
    set_active_llm,
)
from .utils import SIMULATION_YEARS, new_run_id, truncate
from .year_simulation import run_year_cycle
from .monitor.timeline_judge import evaluate_locked_timeline


def _state_from_dict(state_dict: Dict[str, Any]) -> ScenarioState:
    ss = ScenarioState(
        **{k: v for k, v in state_dict.items() if not k.startswith("_")}
    )
    for key, val in state_dict.items():
        if not key.startswith("_") or key == "_llm":
            continue
        if key == "_latest_round_outputs" and isinstance(val, dict):
            setattr(
                ss,
                key,
                {
                    name: AgentOutput(**item) if isinstance(item, dict) else item
                    for name, item in val.items()
                },
            )
        else:
            setattr(ss, key, val)
    return ss


def _state_to_dict(ss: ScenarioState) -> Dict[str, Any]:
    out = ss.model_dump()
    for key in ("_early_stopped", "_latest_round_outputs"):
        if not hasattr(ss, key):
            continue
        val = getattr(ss, key)
        if key == "_latest_round_outputs" and val:
            out[key] = {
                name: (item.model_dump() if isinstance(item, AgentOutput) else item)
                for name, item in val.items()
            }
        else:
            out[key] = val
    return out


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


def _make_year_cycle_node(year: int):
    def year_cycle_node(state: ScenarioState, llm: LLMClient) -> ScenarioState:
        recorder = _rag_recorder(state)
        return run_year_cycle(state, llm, year, recorder)

    year_cycle_node.__name__ = "year_{}_cycle".format(year)
    return year_cycle_node


def timeline_quality_node(state: ScenarioState, llm: LLMClient) -> ScenarioState:
    """Layer 0 gates + LLM judge on the full locked timeline."""
    result = evaluate_locked_timeline(
        state.seed,
        state.scenario_mode,
        state.resolved_timeline,
        state.year_records,
        llm=llm,
    )
    state.timeline_quality = result
    if result.judge and result.judge.pass_quality_bar:
        state.run_metrics.timeline_judge_passed = True
    for w in result.gates.warnings:
        state.run_metrics.citation_warnings.append("timeline: %s" % w)
    if result.judge and not result.judge.pass_quality_bar:
        state.run_metrics.citation_warnings.append(
            "timeline judge: %s" % (result.judge.one_line_verdict or "below quality bar")
        )
    return state


def red_team_node(state: ScenarioState, llm: LLMClient) -> ScenarioState:
    last_summary = state.discussion_rounds[-1] if state.discussion_rounds else None
    if not last_summary and state.year_records:
        rounds = state.year_records[-1].discussion_rounds
        last_summary = rounds[-1] if rounds else None

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
        resolved_timeline=state.resolved_timeline,
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
    if not last_summary and state.year_records:
        rounds = state.year_records[-1].discussion_rounds
        last_summary = rounds[-1] if rounds else None

    last_per_agent: Dict[str, AgentOutput] = {
        name: outs[-1]
        for name, outs in state.agent_outputs.items()
        if outs and name in DOMAIN_AGENTS
    }
    red_team_history = state.agent_outputs.get("red_team") or []
    red_team_last = red_team_history[-1] if red_team_history else AgentOutput(
        agent_name="red_team", main_assessment=""
    )

    state.final_timeline = agent_mod.build_final_timeline(state.resolved_timeline)

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
        resolved_timeline=state.resolved_timeline,
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
            resolved_timeline=state.resolved_timeline,
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

    all_discussions: List[DiscussionSummary] = []
    for record in state.year_records:
        all_discussions.extend(record.discussion_rounds)

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
        discussion_summary=all_discussions or state.discussion_rounds,
        year_records=list(state.year_records),
        timeline_quality=state.timeline_quality,
        image_prompt=state.image_prompt,
        image=state.image_result,
        run_metrics=state.run_metrics,
    )


def _build_nodes() -> List[tuple]:
    nodes: List[tuple] = [
        ("orchestrator_initialize", orchestrator_initialize),
        ("evidence_rag_agent", evidence_rag_node),
    ]
    for year in SIMULATION_YEARS:
        nodes.append(("year_{}_cycle".format(year), _make_year_cycle_node(year)))
    nodes.extend(
        [
            ("timeline_quality_check", timeline_quality_node),
            ("red_team_agent", red_team_node),
            ("orchestrator_synthesis", orchestrator_synthesis),
            ("orchestrator_image_generation", orchestrator_image_generation),
            ("save_run", save_run_node),
        ]
    )
    return nodes


NODES: List[tuple] = _build_nodes()


def build_graph(checkpointer=None, **compile_kwargs):
    try:
        from langgraph.graph import StateGraph, END  # type: ignore
    except Exception:
        return None

    graph = StateGraph(dict)

    def make_wrapper(fn: Callable, name: str):
        def _w(state_dict: Dict[str, Any]) -> Dict[str, Any]:
            llm = get_active_llm()
            if llm is None:
                llm = LLMClient()
            ss = _state_from_dict(state_dict)
            ss = fn(ss, llm)
            return _state_to_dict(ss)

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
    if checkpointer is not None:
        return graph.compile(checkpointer=checkpointer, **compile_kwargs)
    return graph.compile(**compile_kwargs)


def _finalize_run(
    state: ScenarioState,
    llm: LLMClient,
    start: float,
) -> FinalScenario:
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


def _invoke_compiled_graph(
    compiled,
    *,
    run_id: str,
    seed: str,
    scenario_mode: str,
    llm: LLMClient,
    resume: bool,
) -> ScenarioState:
    config = graph_run_config(run_id)
    token = set_active_llm(llm)
    try:
        if resume:
            snapshot = compiled.get_state(config)
            if not snapshot.next:
                values = snapshot.values or {}
                return _state_from_dict(values)
            result = compiled.invoke(None, config=config)
        else:
            state = ScenarioState(
                run_id=run_id,
                seed=seed,
                scenario_mode=scenario_mode,
            )
            state.run_metrics = RunMetrics()
            result = compiled.invoke(_state_to_dict(state), config=config)
        return _state_from_dict(result)
    finally:
        reset_active_llm(token)


def run_graph(
    seed: str,
    scenario_mode: str,
    llm: Optional[LLMClient] = None,
    *,
    run_id: Optional[str] = None,
    resume: bool = False,
) -> FinalScenario:
    llm = llm or LLMClient()
    run_id = run_id or new_run_id()
    start = time.time()

    checkpointer = get_checkpointer()
    compiled = build_graph(checkpointer=checkpointer)

    if compiled is not None:
        try:
            if checkpointer is not None:
                state = _invoke_compiled_graph(
                    compiled,
                    run_id=run_id,
                    seed=seed.strip(),
                    scenario_mode=scenario_mode,
                    llm=llm,
                    resume=resume,
                )
            elif resume:
                raise RuntimeError("checkpointing_disabled")
            else:
                state = ScenarioState(
                    run_id=run_id,
                    seed=seed.strip(),
                    scenario_mode=scenario_mode,
                )
                state.run_metrics = RunMetrics()
                token = set_active_llm(llm)
                try:
                    result = compiled.invoke(_state_to_dict(state))
                    state = _state_from_dict(result)
                finally:
                    reset_active_llm(token)
        except Exception as e:
            if resume:
                raise
            state = ScenarioState(
                run_id=run_id,
                seed=seed.strip(),
                scenario_mode=scenario_mode,
            )
            state.run_metrics = RunMetrics()
            state.errors.append("langgraph_failed: " + str(e))
            state = _run_sequential(state, llm)
    elif resume:
        raise RuntimeError(
            "Cannot resume run " + run_id + ": LangGraph checkpointing unavailable"
        )
    else:
        state = ScenarioState(
            run_id=run_id,
            seed=seed.strip(),
            scenario_mode=scenario_mode,
        )
        state.run_metrics = RunMetrics()
        state = _run_sequential(state, llm)

    return _finalize_run(state, llm, start)


def resume_graph(run_id: str, llm: Optional[LLMClient] = None) -> FinalScenario:
    """Continue an interrupted run from the last LangGraph checkpoint."""
    status = get_checkpoint_status(run_id)
    if status is None:
        raise ValueError("No checkpoint found for run_id: " + run_id)
    if not status.get("can_resume"):
        raise ValueError("Run " + run_id + " is already complete")

    return run_graph(
        seed=str(status.get("seed") or ""),
        scenario_mode=str(status.get("scenario_mode") or "base_case"),
        llm=llm,
        run_id=run_id,
        resume=True,
    )


def _run_sequential(state: ScenarioState, llm: LLMClient) -> ScenarioState:
    for _name, fn in NODES:
        try:
            state = fn(state, llm)
        except Exception as e:
            state.errors.append(_name + "_failed: " + str(e))
    return state
