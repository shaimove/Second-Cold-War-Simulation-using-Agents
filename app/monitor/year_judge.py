"""Layer 1: LLM-as-judge for one locked simulation year."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import ValidationError

from .. import config as _config_mod
from ..agents import format_resolved_timeline
from ..llm import LLMClient
from ..schemas import (
    YEAR_FAILURE_MODE_VALUES,
    YEAR_JUDGE_DIMENSION_NAMES,
    AgentOutput,
    DiscussionSummary,
    JudgeDimension,
    YearBlock,
    YearGateReport,
    YearJudgeVerdict,
)
from ..utils import truncate
from .year_gates import check_year_gates


def build_year_judge_bundle(
    target_year: int,
    seed: str,
    scenario_mode: str,
    year_block: Union[YearBlock, Dict[str, Any]],
    discussion_rounds: List[DiscussionSummary],
    latest_outputs: Dict[str, AgentOutput],
    prior_timeline: Optional[List[YearBlock]] = None,
    year_gates: Optional[YearGateReport] = None,
) -> Dict[str, Any]:
    if isinstance(year_block, YearBlock):
        block_data = year_block.model_dump()
    else:
        block_data = dict(year_block)

    last_disc: Dict[str, Any] = {}
    if discussion_rounds:
        last = discussion_rounds[-1]
        last_disc = {
            "areas_of_disagreement": (last.areas_of_disagreement or [])[:6],
            "key_uncertainties": (last.key_uncertainties or [])[:5],
            "areas_of_agreement": (last.areas_of_agreement or [])[:4],
        }

    agent_positions = {
        name: truncate(out.main_assessment, 350)
        for name, out in latest_outputs.items()
    }

    return {
        "target_year": target_year,
        "seed": seed,
        "scenario_mode": scenario_mode,
        "locked_prior_years": format_resolved_timeline(prior_timeline or []),
        "locked_year_block": block_data,
        "last_discussion": last_disc,
        "agent_positions": agent_positions,
        "layer0_gates": (year_gates.model_dump() if year_gates else {}),
    }


def _compute_pass_bar(verdict: YearJudgeVerdict) -> bool:
    if verdict.overall_score < 3.5:
        return False
    for dim in verdict.dimensions:
        if dim.score <= 1:
            return False
    return True


def _coerce_year_verdict(raw: Dict[str, Any], layer0_blockers: List[str]) -> YearJudgeVerdict:
    dims_raw = raw.get("dimensions") or []
    dimensions: List[JudgeDimension] = []
    if isinstance(dims_raw, list):
        for item in dims_raw:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip().lower()
            if name not in YEAR_JUDGE_DIMENSION_NAMES:
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
    for name in YEAR_JUDGE_DIMENSION_NAMES:
        if name not in present:
            dimensions.append(
                JudgeDimension(name=name, score=3, rationale="Not scored by judge.")
            )
    dimensions.sort(key=lambda d: YEAR_JUDGE_DIMENSION_NAMES.index(d.name))

    failure_modes: List[str] = []
    for fm in raw.get("failure_modes") or []:
        f = str(fm).strip().lower()
        if f in YEAR_FAILURE_MODE_VALUES and f not in failure_modes:
            failure_modes.append(f)

    try:
        overall = float(raw.get("overall_score") or 3.0)
    except (TypeError, ValueError):
        overall = 3.0
    overall = max(1.0, min(5.0, overall))
    if not raw.get("overall_score") and dimensions:
        overall = round(sum(d.score for d in dimensions) / len(dimensions), 2)

    verdict = YearJudgeVerdict(
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


def run_year_quality_judge(
    target_year: int,
    seed: str,
    scenario_mode: str,
    year_block: Union[YearBlock, Dict[str, Any]],
    discussion_rounds: List[DiscussionSummary],
    latest_outputs: Dict[str, AgentOutput],
    prior_timeline: Optional[List[YearBlock]] = None,
    llm: Optional[LLMClient] = None,
    *,
    year_gates: Optional[YearGateReport] = None,
    skip_on_layer0_blockers: bool = False,
) -> YearJudgeVerdict:
    """Layer 0 gates + one LLM call evaluating the locked year."""
    gates = year_gates or check_year_gates(
        target_year,
        year_block,
        discussion_rounds,
        latest_outputs,
        prior_timeline,
    )

    if skip_on_layer0_blockers and gates.blockers:
        return YearJudgeVerdict(
            pass_quality_bar=False,
            layer0_blockers=list(gates.blockers),
            judge_skipped=True,
            judge_skip_reason="Layer 0 blockers: " + "; ".join(gates.blockers),
            one_line_verdict="Year judge skipped due to Layer 0 failures.",
            summary_paragraph=(
                "Deterministic gates failed before the LLM judge ran. "
                "Fix structural issues in the year lock or agent outputs."
            ),
        )

    llm = llm or LLMClient()
    bundle = build_year_judge_bundle(
        target_year,
        seed,
        scenario_mode,
        year_block,
        discussion_rounds,
        latest_outputs,
        prior_timeline,
        year_gates=gates,
    )

    system = (
        "You are an independent Quality Judge for ONE locked year in a multi-agent "
        "USA-China rivalry simulation (2026-2031). Evaluate whether the orchestrator's "
        "year lock faithfully reflects the specialist discussion, respects the seed "
        "and prior locked years, and preserves uncertainty. Scenarios are plausible "
        "planning exercises, not predictions. Reward preserved disagreement; penalize "
        "generic filler and overconfident locks."
    )

    user = (
        "Evaluate this locked simulation year.\n\n"
        + json.dumps(bundle, ensure_ascii=False, indent=2)
        + "\n\nReturn JSON only:\n"
        "{\n"
        '  "overall_score": 4.0,\n'
        '  "dimensions": [\n'
        '    {"name": "year_scope", "score": 4, "rationale": "..."},\n'
        '    {"name": "seed_fidelity", "score": 4, "rationale": "..."},\n'
        '    {"name": "discussion_fidelity", "score": 4, "rationale": "..."},\n'
        '    {"name": "prior_timeline_coherence", "score": 4, "rationale": "..."},\n'
        '    {"name": "uncertainty_preservation", "score": 3, "rationale": "..."}\n'
        "  ],\n"
        '  "failure_modes": ["ignored_agent_disagreement"],\n'
        '  "one_line_verdict": "...",\n'
        '  "summary_paragraph": "3-5 sentences on quality of this year lock."\n'
        "}\n\n"
        "failure_modes must be from: "
        + ", ".join(YEAR_FAILURE_MODE_VALUES)
        + " (or empty list). Include each dimension exactly once."
    )

    fallback = {
        "overall_score": 3.0,
        "dimensions": [
            {"name": n, "score": 3, "rationale": "Year judge fallback."}
            for n in YEAR_JUDGE_DIMENSION_NAMES
        ],
        "failure_modes": [],
        "one_line_verdict": "Year quality judge fallback used.",
        "summary_paragraph": (
            "The year judge could not produce a full assessment; a neutral fallback "
            "score was applied."
        ),
    }

    data = llm.call_llm_json(
        system_prompt=system,
        user_prompt=user,
        agent_name="year_quality_judge",
        round_number=target_year,
        schema_name="year_judge_verdict",
        cache_context={
            "seed": seed,
            "mode": scenario_mode,
            "year": target_year,
        },
        fallback=fallback,
    )

    try:
        verdict = _coerce_year_verdict(
            data if isinstance(data, dict) else fallback,
            layer0_blockers=list(gates.blockers),
        )
    except (ValidationError, TypeError, ValueError):
        verdict = _coerce_year_verdict(fallback, layer0_blockers=list(gates.blockers))

    if gates.warnings and "layer0_warnings" not in verdict.failure_modes:
        pass  # warnings stay on YearGateReport only
    return verdict


def evaluate_locked_year(
    target_year: int,
    seed: str,
    scenario_mode: str,
    year_block: Union[YearBlock, Dict[str, Any]],
    discussion_rounds: List[DiscussionSummary],
    latest_outputs: Dict[str, AgentOutput],
    prior_timeline: Optional[List[YearBlock]] = None,
    llm: Optional[LLMClient] = None,
) -> Tuple[YearGateReport, Optional[YearJudgeVerdict]]:
    """Layer 0 always; Layer 1 LLM when ENABLE_YEAR_JUDGE is true."""
    gates = check_year_gates(
        target_year,
        year_block,
        discussion_rounds,
        latest_outputs,
        prior_timeline,
    )
    if not _config_mod.CONFIG.enable_year_judge:
        return gates, None

    verdict = run_year_quality_judge(
        target_year,
        seed,
        scenario_mode,
        year_block,
        discussion_rounds,
        latest_outputs,
        prior_timeline,
        llm=llm,
        year_gates=gates,
    )
    return gates, verdict
