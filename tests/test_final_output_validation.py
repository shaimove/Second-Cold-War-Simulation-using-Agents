"""Tests for final orchestrator validation / repair pipeline."""
from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest
from pydantic import ValidationError

from app.final_output_validation import (
    _cleanup_timeline_events,
    build_safe_fallback_synthesis,
    coerce_probability,
    deterministic_cleanup,
    normalize_enum_value,
    resolve_orchestrator_synthesis,
    validate_orchestrator_synthesis,
)
from app.llm import LLMClient
from app.schemas import DiscussionSummary, RunMetrics, YearBlock, TimelineEvent


def _valid_payload():
    return {
        "scenario_title": "Tech Rivalry Tightens",
        "scenario_summary": "A plausible five-year path under the seed assumption.",
        "event_status": "hypothetical",
        "key_assumptions": ["No major hot war"],
        "main_disagreements": ["Pace of decoupling"],
        "image_prompt": "Editorial illustration of Pacific rivalry",
    }


def test_valid_output_passes_immediately():
    model, errs = validate_orchestrator_synthesis(_valid_payload())
    assert errs == []
    assert model is not None
    assert model.scenario_title == "Tech Rivalry Tightens"


def test_markdown_wrapped_json_is_cleaned():
    wrapped = "```json\n" + json.dumps(_valid_payload()) + "\n```"
    model, errs = validate_orchestrator_synthesis(wrapped)
    assert errs == []
    assert model is not None
    assert model.event_status == "hypothetical"


def test_string_probability_is_converted_to_float():
    assert coerce_probability("0.45") == 0.45
    assert coerce_probability("45%") == 0.45
    assert coerce_probability("120") == 1.0

    raw = [
        {
            "year": 2026,
            "events": [{"event": "x", "probability": "0.6", "impact": "HIGH"}],
        }
    ]
    cleaned = _cleanup_timeline_events(raw)
    assert cleaned[0]["events"][0]["probability"] == 0.6
    assert cleaned[0]["events"][0]["impact"] == "high"


def test_invalid_enum_is_repaired():
    raw = dict(_valid_payload())
    raw["event_status"] = "HYPOTHETICAL"
    model, errs = validate_orchestrator_synthesis(raw)
    assert errs == []
    assert model is not None
    assert model.event_status == "hypothetical"

    assert normalize_enum_value("impact", "MEDIUM", ("low", "medium", "high"), "medium") == "medium"


def test_missing_field_triggers_repair_agent(monkeypatch):
    llm = LLMClient()
    metrics = RunMetrics()
    calls: List[str] = []

    def fake_repair(*_args, **kwargs):
        calls.append("repair")
        return _valid_payload()

    monkeypatch.setattr(
        "app.final_output_validation.run_json_repair_agent", fake_repair
    )

    raw = {"scenario_summary": "only summary present"}  # missing title
    result, passed, img_off = resolve_orchestrator_synthesis(
        llm=llm,
        raw_output=raw,
        seed="test seed",
        scenario_mode="base_case",
        discussion_summary=None,
        regenerate_fn=lambda: raw,
        partial_timeline=[],
        metrics=metrics,
    )
    assert passed is True
    assert img_off is False
    assert "repair" in calls
    assert metrics.synthesis_repair_attempts >= 1
    assert result["scenario_title"] == "Tech Rivalry Tightens"


def test_repair_failure_triggers_regeneration(monkeypatch):
    llm = LLMClient()
    metrics = RunMetrics()
    regen_calls = {"n": 0}

    def always_bad_repair(*_a, **_k):
        return {"scenario_summary": "still missing title"}

    def regenerate():
        regen_calls["n"] += 1
        return _valid_payload()

    monkeypatch.setattr(
        "app.final_output_validation.run_json_repair_agent", always_bad_repair
    )

    raw = {"scenario_summary": "bad"}
    result, passed, img_off = resolve_orchestrator_synthesis(
        llm=llm,
        raw_output=raw,
        seed="seed",
        scenario_mode="base_case",
        discussion_summary=None,
        regenerate_fn=regenerate,
        partial_timeline=[],
        metrics=metrics,
    )
    assert regen_calls["n"] == 1
    assert metrics.synthesis_regeneration_attempts == 1
    assert passed is True
    assert result["scenario_title"] == "Tech Rivalry Tightens"


def test_total_failure_returns_safe_fallback(monkeypatch):
    llm = LLMClient()
    metrics = RunMetrics()

    monkeypatch.setattr(
        "app.final_output_validation.run_json_repair_agent",
        lambda *_a, **_k: {"scenario_summary": "still invalid"},
    )

    partial = [
        YearBlock(
            year=2026,
            headline="h",
            events=[TimelineEvent(event="e", probability=0.5)],
        )
    ]

    result, passed, img_off = resolve_orchestrator_synthesis(
        llm=llm,
        raw_output={"scenario_summary": "x"},
        seed="my seed",
        scenario_mode="escalation",
        discussion_summary=DiscussionSummary(round_number=1),
        regenerate_fn=lambda: {"scenario_summary": "still invalid"},
        partial_timeline=partial,
        metrics=metrics,
    )
    assert passed is False
    assert img_off is True
    assert metrics.synthesis_used_fallback is True
    assert result.get("image_generation_disabled") is True
    assert result.get("seed") == "my seed"
    assert result.get("scenario_mode") == "escalation"
    assert "error" in result
    assert len(result.get("partial_timeline") or []) == 1


def test_safe_fallback_builder():
    fb = build_safe_fallback_synthesis(
        seed="s",
        scenario_mode="wildcard",
        error_message="boom",
        partial_timeline=[],
    )
    assert fb["error"] == "boom"
    assert fb["image_generation_disabled"] is True


def test_graph_synthesis_applies_validation(monkeypatch):
    """Integration: orchestrator_synthesis node uses validation pipeline."""
    from app.graph import orchestrator_synthesis
    from app.llm import LLMClient
    from app.schemas import ScenarioState, EvidenceSummary, AgentOutput

    state = ScenarioState(
        run_id="run_test",
        seed="test",
        scenario_mode="base_case",
        evidence_summary=EvidenceSummary(),
        agent_outputs={
            "geo_strategy": [
                AgentOutput(agent_name="geo_strategy", main_assessment="a")
            ],
        },
        discussion_rounds=[],
        red_team_findings=[],
    )
    llm = LLMClient()

    monkeypatch.setattr(
        "app.agents.run_orchestrator_final_synthesis_raw",
        lambda *a, **k: _valid_payload(),
    )

    state = orchestrator_synthesis(state, llm)
    assert state.run_metrics.synthesis_validation_passed is True
    assert state.scenario_title == "Tech Rivalry Tightens"
