from app.llm import LLMClient
from app.monitor.judge import build_judge_bundle, run_quality_judge
from app.monitor.service import attach_monitor, run_monitor
from app.schemas import JUDGE_DIMENSION_NAMES


def test_build_judge_bundle_compact():
    bundle = build_judge_bundle(
        {
            "seed": "chip controls",
            "scenario_mode": "base_case",
            "scenario_title": "T",
            "scenario_summary": "S" * 2000,
            "timeline": [{"year": 2026, "headline": "h", "events": []}],
            "agent_summaries": {"geo_strategy": "x"},
            "run_metrics": {"unique_chunks_used": 2},
        }
    )
    assert bundle["seed"] == "chip controls"
    assert len(bundle["scenario_summary"]) <= 1200
    assert "pipeline" in bundle
    assert "image_prompt" not in bundle


def test_quality_judge_mock_mode():
    llm = LLMClient()
    verdict = run_quality_judge(
        {
            "seed": "test",
            "scenario_mode": "base_case",
            "scenario_title": "Title",
            "scenario_summary": "Summary",
            "timeline": [{"year": 2026, "headline": "h"}],
            "agent_summaries": {"geo_strategy": "a"},
            "run_metrics": {},
        },
        llm=llm,
    )
    assert verdict.overall_score >= 1.0
    assert len(verdict.dimensions) == 5
    assert {d.name for d in verdict.dimensions} == set(JUDGE_DIMENSION_NAMES)
    assert len(verdict.summary_paragraph) > 20


def test_run_monitor_skips_judge_on_blockers():
    monitor = run_monitor(
        {
            "run_id": "x",
            "seed": "s",
            "scenario_mode": "base_case",
            "scenario_title": "",
            "scenario_summary": "",
            "timeline": [],
            "run_metrics": {"synthesis_used_fallback": True},
        }
    )
    assert monitor.judge_skipped is True
    assert monitor.judge is None


def test_run_monitor_gates_only_skips_judge():
    monitor = run_monitor(
        {
            "run_id": "r1",
            "seed": "Taiwan test",
            "scenario_mode": "base_case",
            "scenario_title": "T",
            "scenario_summary": "S",
            "timeline": [{"year": y, "headline": "h"} for y in range(2026, 2032)],
            "main_disagreements": ["a"],
            "discussion_summary": [{"round_number": 1, "areas_of_disagreement": ["a"]}],
            "run_metrics": {"synthesis_validation_passed": True},
        },
        skip_judge=True,
    )
    assert monitor.gates.passed is True
    assert monitor.judge is None
    assert monitor.judge_skipped is True
    assert len(monitor.gates.checks) == 6


def test_attach_monitor_to_payload():
    payload = {"run_id": "r1", "seed": "s"}
    monitor = run_monitor(
        {
            "run_id": "r1",
            "seed": "Taiwan test",
            "scenario_mode": "base_case",
            "scenario_title": "T",
            "scenario_summary": "S",
            "timeline": [{"year": y, "headline": "h"} for y in range(2026, 2032)],
            "main_disagreements": ["a"],
            "discussion_summary": [{"round_number": 1, "areas_of_disagreement": ["a"]}],
            "run_metrics": {"synthesis_validation_passed": True},
        }
    )
    out = attach_monitor(payload, monitor)
    assert "monitor" in out
    assert out["monitor"]["gates"]["passed"] is True
