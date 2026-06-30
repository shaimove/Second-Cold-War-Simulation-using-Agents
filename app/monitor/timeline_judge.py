"""Layer 1: LLM-as-judge on the full locked timeline (after year 2031)."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union

from pydantic import ValidationError

from .. import config as _config_mod
from ..agents import format_resolved_timeline
from ..llm import LLMClient
from ..schemas import (
    TIMELINE_FAILURE_MODE_VALUES,
    TIMELINE_JUDGE_DIMENSION_NAMES,
    GateReport,
    JudgeDimension,
    TimelineJudgeVerdict,
    TimelineQualityResult,
    YearBlock,
    YearSimulationRecord,
)
from ..utils import truncate, utcnow_iso
from .timeline_gates import check_timeline_gates


def build_timeline_judge_bundle(
    seed: str,
    scenario_mode: str,
    timeline: List[Union[YearBlock, Dict[str, Any]]],
    year_records: Optional[List[Union[YearSimulationRecord, dict]]] = None,
    timeline_gates: Optional[GateReport] = None,
) -> Dict[str, Any]:
    timeline_headlines: List[str] = []
    for yb in timeline:
        if isinstance(yb, dict):
            year = yb.get("year", "?")
            headline = (yb.get("headline") or "").strip()
            events = yb.get("events") or []
        else:
            year = yb.year
            headline = (yb.headline or "").strip()
            events = yb.events
        if not headline and events:
            headline = str((events[0].event if hasattr(events[0], "event") else events[0].get("event")) or "")
        timeline_headlines.append("%s: %s" % (year, truncate(headline, 120)))

    year_judge_summary: List[Dict[str, Any]] = []
    for rec in year_records or []:
        if isinstance(rec, dict):
            year = rec.get("year")
            yj = rec.get("year_judge")
        else:
            year = rec.year
            yj = rec.year_judge
        if not yj:
            continue
        if isinstance(yj, dict):
            year_judge_summary.append(
                {
                    "year": year,
                    "pass_quality_bar": yj.get("pass_quality_bar"),
                    "overall_score": yj.get("overall_score"),
                    "one_line_verdict": yj.get("one_line_verdict"),
                }
            )
        else:
            year_judge_summary.append(
                {
                    "year": year,
                    "pass_quality_bar": yj.pass_quality_bar,
                    "overall_score": yj.overall_score,
                    "one_line_verdict": yj.one_line_verdict,
                }
            )

    return {
        "seed": seed,
        "scenario_mode": scenario_mode,
        "locked_timeline": format_resolved_timeline(
            [YearBlock(**yb) if isinstance(yb, dict) else yb for yb in timeline]
        ),
        "timeline_headlines": timeline_headlines,
        "per_year_judges": year_judge_summary,
        "layer0_gates": (timeline_gates.model_dump() if timeline_gates else {}),
    }


def _compute_pass_bar(verdict: TimelineJudgeVerdict) -> bool:
    if verdict.overall_score < 3.5:
        return False
    for dim in verdict.dimensions:
        if dim.score <= 1:
            return False
    return True


def _coerce_timeline_verdict(
    raw: Dict[str, Any],
    layer0_blockers: List[str],
) -> TimelineJudgeVerdict:
    dims_raw = raw.get("dimensions") or []
    dimensions: List[JudgeDimension] = []
    if isinstance(dims_raw, list):
        for item in dims_raw:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip().lower()
            if name not in TIMELINE_JUDGE_DIMENSION_NAMES:
                continue
            try:
                score = int(item.get("score") or 3)
            except (TypeError, ValueError):
                score = 3
            score = max(1, min(5, score))
            dimensions.append(
                JudgeDimension(
                    name=name,
                    score=score,
                    rationale=str(item.get("rationale") or "")[:300],
                )
            )

    present = {d.name for d in dimensions}
    for name in TIMELINE_JUDGE_DIMENSION_NAMES:
        if name not in present:
            dimensions.append(
                JudgeDimension(name=name, score=3, rationale="Not scored by judge.")
            )
    dimensions.sort(key=lambda d: TIMELINE_JUDGE_DIMENSION_NAMES.index(d.name))

    failure_modes: List[str] = []
    for fm in raw.get("failure_modes") or []:
        f = str(fm).strip().lower()
        if f in TIMELINE_FAILURE_MODE_VALUES and f not in failure_modes:
            failure_modes.append(f)

    try:
        overall = float(raw.get("overall_score") or 3.0)
    except (TypeError, ValueError):
        overall = 3.0
    overall = max(1.0, min(5.0, overall))
    if not raw.get("overall_score") and dimensions:
        overall = round(sum(d.score for d in dimensions) / len(dimensions), 2)

    verdict = TimelineJudgeVerdict(
        overall_score=overall,
        pass_quality_bar=False,
        dimensions=dimensions,
        failure_modes=failure_modes,
        one_line_verdict=str(raw.get("one_line_verdict") or "")[:200],
        summary_paragraph=str(raw.get("summary_paragraph") or "")[:900],
        layer0_blockers=list(layer0_blockers),
    )
    verdict.pass_quality_bar = _compute_pass_bar(verdict)
    return verdict


def run_timeline_quality_judge(
    seed: str,
    scenario_mode: str,
    timeline: List[Union[YearBlock, Dict[str, Any]]],
    year_records: Optional[List[Union[YearSimulationRecord, dict]]] = None,
    llm: Optional[LLMClient] = None,
    *,
    timeline_gates: Optional[GateReport] = None,
) -> TimelineJudgeVerdict:
    gates = timeline_gates or check_timeline_gates(timeline, year_records)
    llm = llm or LLMClient()
    bundle = build_timeline_judge_bundle(
        seed,
        scenario_mode,
        timeline,
        year_records,
        timeline_gates=gates,
    )

    system = (
        "You are an independent Quality Judge for a locked six-year simulation "
        "timeline (2026-2031) in a USA-China rivalry scenario planner. Evaluate "
        "whether the full arc is coherent, seed-faithful, and useful as fixed "
        "history for red-team critique and final synthesis. This is a plausible "
        "scenario exercise, not a prediction."
    )

    user = (
        "Evaluate this locked timeline.\n\n"
        + json.dumps(bundle, ensure_ascii=False, indent=2)
        + "\n\nReturn JSON only:\n"
        "{\n"
        '  "overall_score": 4.0,\n'
        '  "dimensions": [\n'
        '    {"name": "timeline_completeness", "score": 4, "rationale": "..."},\n'
        '    {"name": "seed_fidelity", "score": 4, "rationale": "..."},\n'
        '    {"name": "cross_year_coherence", "score": 4, "rationale": "..."},\n'
        '    {"name": "escalation_arc", "score": 3, "rationale": "..."},\n'
        '    {"name": "uncertainty_preservation", "score": 3, "rationale": "..."}\n'
        "  ],\n"
        '  "failure_modes": ["overconfident_timeline"],\n'
        '  "one_line_verdict": "...",\n'
        '  "summary_paragraph": "3-5 sentences on timeline quality."\n'
        "}\n\n"
        "failure_modes must be from: "
        + ", ".join(TIMELINE_FAILURE_MODE_VALUES)
        + " (or empty list). Include each dimension exactly once."
    )

    fallback = {
        "overall_score": 3.0,
        "dimensions": [
            {"name": n, "score": 3, "rationale": "Timeline judge fallback."}
            for n in TIMELINE_JUDGE_DIMENSION_NAMES
        ],
        "failure_modes": [],
        "one_line_verdict": "Timeline quality judge fallback used.",
        "summary_paragraph": (
            "The timeline judge could not produce a full assessment; a neutral "
            "fallback score was applied."
        ),
    }

    data = llm.call_llm_json(
        system_prompt=system,
        user_prompt=user,
        agent_name="timeline_quality_judge",
        round_number=0,
        schema_name="timeline_judge_verdict",
        cache_context={"seed": seed, "mode": scenario_mode},
        fallback=fallback,
    )

    try:
        return _coerce_timeline_verdict(
            data if isinstance(data, dict) else fallback,
            layer0_blockers=list(gates.blockers),
        )
    except (ValidationError, TypeError, ValueError):
        return _coerce_timeline_verdict(fallback, layer0_blockers=list(gates.blockers))


def evaluate_locked_timeline(
    seed: str,
    scenario_mode: str,
    timeline: List[Union[YearBlock, Dict[str, Any]]],
    year_records: Optional[List[Union[YearSimulationRecord, dict]]] = None,
    llm: Optional[LLMClient] = None,
) -> TimelineQualityResult:
    """Layer 0 always; Layer 1 LLM when ENABLE_TIMELINE_JUDGE is true."""
    gates = check_timeline_gates(timeline, year_records)
    cfg = _config_mod.CONFIG
    judge_model = cfg.openai_judge_model

    if not cfg.enable_timeline_judge:
        return TimelineQualityResult(
            gates=gates,
            judge_skipped=True,
            judge_skip_reason="Timeline judge disabled (ENABLE_TIMELINE_JUDGE=false).",
            judged_at=utcnow_iso(),
            judge_model=judge_model,
        )

    verdict = run_timeline_quality_judge(
        seed,
        scenario_mode,
        timeline,
        year_records,
        llm=llm,
        timeline_gates=gates,
    )
    return TimelineQualityResult(
        gates=gates,
        judge=verdict,
        judged_at=utcnow_iso(),
        judge_model=judge_model,
    )
