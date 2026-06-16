"""LLM-as-judge: one structured quality call per run."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union

from pydantic import ValidationError

from .. import config as _config_mod
from ..llm import LLMClient
from ..schemas import (
    FAILURE_MODE_VALUES,
    JUDGE_DIMENSION_NAMES,
    FinalScenario,
    JudgeDimension,
    JudgeVerdict,
)
from ..utils import truncate, utcnow_iso


def build_judge_bundle(final: Union[FinalScenario, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(final, dict):
        data = final
    else:
        data = final.model_dump()

    discussion = data.get("discussion_summary") or []
    last_disc: Dict[str, Any] = {}
    if discussion:
        last = discussion[-1]
        if isinstance(last, dict):
            last_disc = {
                "areas_of_disagreement": (last.get("areas_of_disagreement") or [])[:6],
                "key_uncertainties": (last.get("key_uncertainties") or [])[:5],
                "areas_of_agreement": (last.get("areas_of_agreement") or [])[:4],
            }

    timeline_headlines: List[str] = []
    for yb in data.get("timeline") or []:
        year = yb.get("year", "?")
        headline = (yb.get("headline") or "").strip()
        if not headline and yb.get("events"):
            headline = str((yb["events"][0] or {}).get("event") or "")
        timeline_headlines.append("%s: %s" % (year, truncate(headline, 120)))

    metrics = data.get("run_metrics") or {}
    return {
        "seed": data.get("seed", ""),
        "scenario_mode": data.get("scenario_mode", ""),
        "scenario_title": data.get("scenario_title", ""),
        "scenario_summary": truncate(data.get("scenario_summary") or "", 1200),
        "event_status": data.get("event_status", ""),
        "main_disagreements": (data.get("main_disagreements") or [])[:8],
        "key_assumptions": (data.get("key_assumptions") or [])[:8],
        "red_team_warnings": (data.get("red_team_warnings") or [])[:6],
        "agent_summaries": {
            k: truncate(v, 400)
            for k, v in (data.get("agent_summaries") or {}).items()
        },
        "last_discussion": last_disc,
        "timeline_headlines": timeline_headlines,
        "pipeline": {
            "synthesis_used_fallback": bool(metrics.get("synthesis_used_fallback")),
            "synthesis_validation_passed": metrics.get("synthesis_validation_passed"),
            "citation_warnings": (metrics.get("citation_warnings") or [])[:5],
            "unique_chunks_used": int(metrics.get("unique_chunks_used") or 0),
            "discussion_rounds_completed": int(
                metrics.get("discussion_rounds_completed") or 0
            ),
        },
    }


def _compute_pass_bar(verdict: JudgeVerdict) -> bool:
    if verdict.overall_score < 3.5:
        return False
    for dim in verdict.dimensions:
        if dim.score <= 1:
            return False
    return True


def _coerce_verdict(raw: Dict[str, Any]) -> JudgeVerdict:
    dims_raw = raw.get("dimensions") or []
    dimensions: List[JudgeDimension] = []
    if isinstance(dims_raw, list):
        for item in dims_raw:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip().lower()
            if name not in JUDGE_DIMENSION_NAMES:
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

    # Ensure all five dimensions present (fill missing with neutral 3)
    present = {d.name for d in dimensions}
    for name in JUDGE_DIMENSION_NAMES:
        if name not in present:
            dimensions.append(
                JudgeDimension(name=name, score=3, rationale="Not scored by judge.")
            )
    dimensions.sort(key=lambda d: JUDGE_DIMENSION_NAMES.index(d.name))

    failure_modes: List[str] = []
    for fm in raw.get("failure_modes") or []:
        f = str(fm).strip().lower()
        if f in FAILURE_MODE_VALUES and f not in failure_modes:
            failure_modes.append(f)

    try:
        overall = float(raw.get("overall_score") or 3.0)
    except (TypeError, ValueError):
        overall = 3.0
    overall = max(1.0, min(5.0, overall))
    if not raw.get("overall_score") and dimensions:
        overall = round(sum(d.score for d in dimensions) / len(dimensions), 2)

    verdict = JudgeVerdict(
        overall_score=overall,
        pass_quality_bar=False,
        dimensions=dimensions,
        failure_modes=failure_modes,
        one_line_verdict=str(raw.get("one_line_verdict") or "")[:200],
        summary_paragraph=str(raw.get("summary_paragraph") or "")[:900],
    )
    verdict.pass_quality_bar = _compute_pass_bar(verdict)
    return verdict


def run_quality_judge(
    final: Union[FinalScenario, Dict[str, Any]],
    llm: Optional[LLMClient] = None,
) -> JudgeVerdict:
    """One LLM call; validates/coerces to JudgeVerdict."""
    llm = llm or LLMClient()
    bundle = build_judge_bundle(final)

    system = (
        "You are an independent Quality Judge for a multi-agent geopolitical "
        "scenario simulator (USA–China rivalry, 2026–2031). Score the FINAL "
        "artifact only. Scenarios must be plausible planning exercises, not "
        "predictions. Reward preserved disagreement and distinct specialist voices."
    )

    user = (
        "Evaluate this scenario output.\n\n"
        + json.dumps(bundle, ensure_ascii=False, indent=2)
        + "\n\nReturn JSON only:\n"
        "{\n"
        '  "overall_score": 4.2,\n'
        '  "dimensions": [\n'
        '    {"name": "seed_fidelity", "score": 4, "rationale": "..."},\n'
        '    {"name": "plausibility", "score": 4, "rationale": "..."},\n'
        '    {"name": "specialist_diversity", "score": 3, "rationale": "..."},\n'
        '    {"name": "disagreement_preservation", "score": 4, "rationale": "..."},\n'
        '    {"name": "timeline_usefulness", "score": 4, "rationale": "..."}\n'
        "  ],\n"
        '  "failure_modes": ["economic_monoculture"],\n'
        '  "one_line_verdict": "...",\n'
        '  "summary_paragraph": "One paragraph (3-5 sentences) explaining overall '
        'quality, strengths, and main weaknesses for a strategist reading this scenario."\n'
        "}\n\n"
        "failure_modes must be from: "
        + ", ".join(FAILURE_MODE_VALUES)
        + " (or empty list). Include each dimension exactly once."
    )

    fallback = {
        "overall_score": 3.0,
        "dimensions": [
            {"name": n, "score": 3, "rationale": "Judge fallback."}
            for n in JUDGE_DIMENSION_NAMES
        ],
        "failure_modes": [],
        "one_line_verdict": "Quality judge fallback used.",
        "summary_paragraph": (
            "The quality judge could not produce a full assessment; a neutral "
            "fallback score was applied. Re-run the judge when the API is available."
        ),
    }

    data = llm.call_llm_json(
        system_prompt=system,
        user_prompt=user,
        agent_name="quality_judge",
        round_number=0,
        schema_name="judge_verdict",
        cache_context={"seed": bundle.get("seed"), "run": bundle.get("scenario_title")},
        fallback=fallback,
    )

    try:
        return _coerce_verdict(data if isinstance(data, dict) else fallback)
    except (ValidationError, TypeError, ValueError):
        return _coerce_verdict(fallback)
