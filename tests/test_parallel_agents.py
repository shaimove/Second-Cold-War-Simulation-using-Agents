from app.graph import run_graph
from app.llm import LLMClient
from app.parallel_agents import run_domain_agents_parallel
from app.rag import RagMetricsRecorder
from app.schemas import DOMAIN_AGENTS, EvidenceLanes, RunMetrics, ScenarioState


def test_parallel_domain_agents_returns_all_five():
    llm = LLMClient()
    state = ScenarioState(run_id="p1", seed="Taiwan election scenario", scenario_mode="base_case", current_year=2026)
    state.evidence_lanes = EvidenceLanes(observed_blob="test lane")
    recorder = RagMetricsRecorder(RunMetrics())

    latest = run_domain_agents_parallel(
        state, llm, round_number=1, recorder=recorder, previous_summary=None, disagreement_chunks=[]
    )
    assert set(latest.keys()) == set(DOMAIN_AGENTS)
    for name in DOMAIN_AGENTS:
        assert len(state.agent_outputs[name]) == 1


def test_graph_parallel_mode_end_to_end():
    final = run_graph(seed="Taiwan and chip export controls", scenario_mode="base_case")
    for name in DOMAIN_AGENTS:
        assert name in final.run_metrics.agents_used
