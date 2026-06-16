"""Deterministic hard gates before LLM judge (no API cost)."""
from __future__ import annotations

from typing import Any, Dict, List, Union

from ..schemas import FinalScenario, GateCheck, GateReport

GATE_LABELS = {
    "G1": "Structural completeness",
    "G2": "Synthesis fallback",
    "G3": "Synthesis validation",
    "G4": "Pipeline health",
    "G5": "Disagreement preservation",
    "G6": "Discussion / repair health",
}


def _check(
    gate_id: str,
    *,
    fail_detail: str = "",
    warn_detail: str = "",
) -> GateCheck:
    if fail_detail:
        return GateCheck(
            id=gate_id,
            label=GATE_LABELS[gate_id],
            status="fail",
            detail=fail_detail,
        )
    if warn_detail:
        return GateCheck(
            id=gate_id,
            label=GATE_LABELS[gate_id],
            status="warn",
            detail=warn_detail,
        )
    return GateCheck(
        id=gate_id,
        label=GATE_LABELS[gate_id],
        status="pass",
        detail="OK",
    )


def check_hard_gates(
    final: Union[FinalScenario, Dict[str, Any]],
) -> GateReport:
    """Return blockers (skip judge), warnings (judge anyway), and per-gate checks."""
    if isinstance(final, dict):
        data = final
    else:
        data = final.model_dump()

    blockers: List[str] = []
    warnings: List[str] = []
    checks: List[GateCheck] = []

    timeline = data.get("timeline") or []
    title = (data.get("scenario_title") or "").strip()
    summary = (data.get("scenario_summary") or "").strip()

    # G1 — structural completeness
    g1_fail_parts: List[str] = []
    g1_warn = ""
    if not title:
        g1_fail_parts.append("missing scenario_title")
    if not summary:
        g1_fail_parts.append("missing scenario_summary")
    if len(timeline) != 6:
        g1_fail_parts.append(
            "timeline must have 6 year blocks (got %d)" % len(timeline)
        )
    else:
        empty_years = [
            yb.get("year")
            for yb in timeline
            if not (yb.get("headline") or "").strip()
            and not (yb.get("events") or [])
        ]
        if len(empty_years) >= 4:
            g1_warn = "%d/6 timeline years have no headline or events" % len(
                empty_years
            )

    if g1_fail_parts:
        detail = "; ".join(g1_fail_parts)
        blockers.append("G1: " + detail)
        checks.append(_check("G1", fail_detail=detail))
    elif g1_warn:
        warnings.append("G1: " + g1_warn)
        checks.append(_check("G1", warn_detail=g1_warn))
    else:
        checks.append(_check("G1"))

    metrics = data.get("run_metrics") or {}

    # G2 — synthesis fallback
    if metrics.get("synthesis_used_fallback"):
        detail = "synthesis_used_fallback is true"
        blockers.append("G2: " + detail)
        checks.append(_check("G2", fail_detail=detail))
    else:
        checks.append(_check("G2"))

    # G3 — synthesis validation
    if metrics.get("synthesis_validation_passed") is False:
        detail = "synthesis_validation_passed is false"
        blockers.append("G3: " + detail)
        checks.append(_check("G3", fail_detail=detail))
    else:
        checks.append(_check("G3"))

    # G4 — pipeline errors (critical node failures)
    g4_fail = ""
    for err in data.get("errors") or []:
        if not err:
            continue
        if any(
            tok in str(err).lower()
            for tok in ("_failed", "langgraph_failed", "save_failed")
        ):
            g4_fail = str(err)[:120]
            blockers.append("G4: pipeline error — " + g4_fail)
            break
    if g4_fail:
        checks.append(_check("G4", fail_detail=g4_fail))
    else:
        checks.append(_check("G4"))

    # G5 — false consensus
    main_dis = [d for d in (data.get("main_disagreements") or []) if str(d).strip()]
    discussion = data.get("discussion_summary") or []
    last_dis: List[str] = []
    if discussion:
        last = discussion[-1]
        if isinstance(last, dict):
            last_dis = last.get("areas_of_disagreement") or []
    if not main_dis and not last_dis:
        detail = "no disagreements in final output or last discussion round"
        warnings.append("G5: " + detail)
        checks.append(_check("G5", warn_detail=detail))
    else:
        checks.append(_check("G5"))

    # G6 — red-team / rounds health
    g6_warn = ""
    if int(metrics.get("discussion_rounds_completed") or 0) < 1:
        g6_warn = "no discussion rounds completed"
    elif int(metrics.get("synthesis_repair_attempts") or 0) > 0:
        g6_warn = "synthesis required %d repair attempts" % int(
            metrics["synthesis_repair_attempts"]
        )
    if g6_warn:
        warnings.append("G6: " + g6_warn)
        checks.append(_check("G6", warn_detail=g6_warn))
    else:
        checks.append(_check("G6"))

    passed = len(blockers) == 0
    return GateReport(
        passed=passed,
        blockers=blockers,
        warnings=warnings,
        checks=checks,
    )
