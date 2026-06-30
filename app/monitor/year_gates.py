"""Layer 0: deterministic checks on a locked simulation year (no LLM)."""
from __future__ import annotations

from typing import Dict, List, Optional, Union

from ..schemas import (
    AgentOutput,
    DiscussionSummary,
    DOMAIN_AGENTS,
    GateCheck,
    YearBlock,
    YearGateReport,
)

YEAR_GATE_LABELS = {
    "Y1": "Year block structure",
    "Y2": "Discussion ran",
    "Y3": "Agent coverage",
    "Y4": "Year scope in agent outputs",
    "Y5": "Prior timeline ordering",
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
            label=YEAR_GATE_LABELS[gate_id],
            status="fail",
            detail=fail_detail,
        )
    if warn_detail:
        return GateCheck(
            id=gate_id,
            label=YEAR_GATE_LABELS[gate_id],
            status="warn",
            detail=warn_detail,
        )
    return GateCheck(
        id=gate_id,
        label=YEAR_GATE_LABELS[gate_id],
        status="pass",
        detail="OK",
    )


def check_year_gates(
    target_year: int,
    year_block: Union[YearBlock, dict],
    discussion_rounds: List[DiscussionSummary],
    latest_outputs: Dict[str, AgentOutput],
    prior_timeline: Optional[List[YearBlock]] = None,
) -> YearGateReport:
    """Free checks after orchestrator locks one year."""
    if isinstance(year_block, dict):
        block = YearBlock(**year_block)
    else:
        block = year_block

    blockers: List[str] = []
    warnings: List[str] = []
    checks: List[GateCheck] = []

    # Y1 — structure
    y1_fail: List[str] = []
    if block.year != target_year:
        y1_fail.append("year field %d != target %d" % (block.year, target_year))
    if not (block.headline or "").strip():
        y1_fail.append("missing headline")
    if not (block.headline or "").strip() and not block.events:
        y1_fail.append("no headline or events")
    checks.append(_check("Y1", fail_detail="; ".join(y1_fail)))
    if y1_fail:
        blockers.append("Y1: " + "; ".join(y1_fail))

    # Y2 — discussion ran
    y2_fail = ""
    if not discussion_rounds:
        y2_fail = "no discussion rounds recorded"
    checks.append(_check("Y2", fail_detail=y2_fail))
    if y2_fail:
        blockers.append("Y2: " + y2_fail)

    # Y3 — all five domain agents contributed
    missing = [name for name in DOMAIN_AGENTS if name not in latest_outputs]
    y3_fail = ""
    y3_warn = ""
    if missing:
        y3_fail = "missing agent outputs: " + ", ".join(missing)
    elif any(not (latest_outputs[n].main_assessment or "").strip() for n in DOMAIN_AGENTS):
        y3_warn = "one or more agents returned empty main_assessment"
    checks.append(_check("Y3", fail_detail=y3_fail, warn_detail=y3_warn))
    if y3_fail:
        blockers.append("Y3: " + y3_fail)
    if y3_warn:
        warnings.append("Y3: " + y3_warn)

    # Y4 — agent timeline contributions scoped to target year
    wrong_year: List[str] = []
    for name, out in latest_outputs.items():
        for tc in out.timeline_contributions:
            if tc.year and tc.year != target_year:
                wrong_year.append("%s contributed event for %d" % (name, tc.year))
    y4_warn = "; ".join(wrong_year[:4])
    if len(wrong_year) > 4:
        y4_warn += " (+%d more)" % (len(wrong_year) - 4)
    checks.append(_check("Y4", warn_detail=y4_warn))
    if y4_warn:
        warnings.append("Y4: " + y4_warn)

    # Y5 — prior timeline must be strictly before this year
    y5_fail = ""
    y5_warn = ""
    prior = prior_timeline or []
    for prev in prior:
        if prev.year >= target_year:
            y5_fail = "prior locked year %d is not before target %d" % (
                prev.year,
                target_year,
            )
            break
    if not y5_fail and len({p.year for p in prior}) != len(prior):
        y5_warn = "duplicate years in prior locked timeline"
    checks.append(_check("Y5", fail_detail=y5_fail, warn_detail=y5_warn))
    if y5_fail:
        blockers.append("Y5: " + y5_fail)
    if y5_warn:
        warnings.append("Y5: " + y5_warn)

    passed = len(blockers) == 0
    return YearGateReport(
        passed=passed,
        blockers=blockers,
        warnings=warnings,
        checks=checks,
    )
