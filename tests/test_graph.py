from app.graph import run_graph
from app.llm import LLMClient
from app.schemas import DOMAIN_AGENTS


def test_graph_runs_end_to_end_in_mock_mode():
    llm = LLMClient()
    final = run_graph(seed="If Taiwan elects an independence-leaning government.", scenario_mode="base_case", llm=llm)

    assert final.run_id
    assert final.scenario_title
    assert final.scenario_summary
    assert len(final.timeline) == 6
    for yb, expected in zip(final.timeline, [2026, 2027, 2028, 2029, 2030, 2031]):
        assert yb.year == expected
    # Agents present in metrics include all domain agents + orchestrator + red_team + evidence.
    used = set(final.run_metrics.agents_used)
    for name in DOMAIN_AGENTS:
        assert name in used
    assert "evidence_rag" in used
    assert "red_team" in used
    assert final.run_metrics.years_completed == 6
    assert final.run_metrics.year_judges_run == 6
    assert final.run_metrics.year_judges_passed == 6
    assert len(final.year_records) == 6
    assert all(r.year_judge is not None for r in final.year_records)
    assert all(r.year_gates is not None for r in final.year_records)
    assert final.timeline_quality is not None
    assert final.timeline_quality.gates.passed is True
    assert final.run_metrics.timeline_judge_passed is True


def test_graph_runs_all_three_rounds_when_configured():
    final = run_graph(seed="x", scenario_mode="escalation")
    assert final.run_metrics.years_completed == 6
    assert final.run_metrics.year_judges_run == 6
    assert final.run_metrics.year_judges_passed == 6
    assert len(final.year_records) == 6
    assert all(r.year_judge is not None for r in final.year_records)
    assert all(r.year_gates is not None for r in final.year_records)
    assert final.timeline_quality is not None
    assert final.timeline_quality.gates.passed is True
    assert final.run_metrics.timeline_judge_passed is True


def test_graph_image_generation_does_not_crash_run():
    final = run_graph(seed="x", scenario_mode="wildcard")
    assert final.image.enabled is True
    # In mock mode we write a placeholder PNG.
    assert final.image.generated is True
    assert final.image.path is not None


def test_graph_with_each_mode_runs():
    for mode in ("base_case", "escalation", "de_escalation", "wildcard"):
        final = run_graph(seed="seed", scenario_mode=mode)
        assert final.scenario_mode == mode
