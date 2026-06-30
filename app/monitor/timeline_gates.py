"""Layer 0: deterministic checks on the full locked 2026-2031 timeline."""
from __future__ import annotations

from typing import List, Optional, Union

from ..schemas import (
    GateCheck,
    GateReport,
    YearBlock,
    YearSimulationRecord,
)
from ..utils import SIMULATION_YEARS

TIMELINE_GATE_LABELS = {
    "T1": "Six-year structure",
    "T2": "Year sequence",
    "T3": "Year records complete",
    "T4": "Per-year judge health",
    "T5": "Per-year gate health",
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
            label=TIMELINE_GATE_LABELS[gate_id],
            status="fail",
            detail=fail_detail,
        )
    if warn_detail:
        return GateCheck(
            id=gate_id,
            label=TIMELINE_GATE_LABELS[gate_id],
            status="warn",
            detail=warn_detail,
        )
    return GateCheck(
        id=gate_id,
        label=TIMELINE_GATE_LABELS[gate_id],
        status="pass",
        detail="OK",
    )


def check_timeline_gates(
    timeline: List[Union[YearBlock, dict]],
    year_records: Optional[List[Union[YearSimulationRecord, dict]]] = None,
) -> GateReport:
    """Free checks after all simulation years are locked."""
    blocks: List[YearBlock] = []
    for item in timeline:
        if isinstance(item, dict):
            blocks.append(YearBlock(**item))
        else:
            blocks.append(item)

    blockers: List[str] = []
    warnings: List[str] = []
    checks: List[GateCheck] = []

    # T1 — six years with headlines
    t1_fail: List[str] = []
    if len(blocks) != 6:
        t1_fail.append("expected 6 year blocks, got %d" % len(blocks))
    empty = [b.year for b in blocks if not (b.headline or "").strip()]
    if empty:
        t1_fail.append("missing headline for years: " + ", ".join(str(y) for y in empty))
    checks.append(_check("T1", fail_detail="; ".join(t1_fail)))
    if t1_fail:
        blockers.append("T1: " + "; ".join(t1_fail))

    # T2 — correct years in order
    t2_fail = ""
    years = [b.year for b in blocks]
    if years != list(SIMULATION_YEARS):
        t2_fail = "years %s != expected %s" % (years, list(SIMULATION_YEARS))
    checks.append(_check("T2", fail_detail=t2_fail))
    if t2_fail:
        blockers.append("T2: " + t2_fail)

    # T3 — year records
    t3_fail = ""
    records = year_records or []
    if len(records) != 6:
        t3_fail = "expected 6 year_records, got %d" % len(records)
    checks.append(_check("T3", fail_detail=t3_fail))
    if t3_fail:
        blockers.append("T3: " + t3_fail)

    # T4 — per-year judge pass rate
    t4_warn = ""
    judged = 0
    passed = 0
    for rec in records:
        if isinstance(rec, dict):
            yj = rec.get("year_judge")
        else:
            yj = rec.year_judge
        if not yj:
            continue
        judged += 1
        if isinstance(yj, dict):
            ok = bool(yj.get("pass_quality_bar"))
        else:
            ok = bool(yj.pass_quality_bar)
        if ok:
            passed += 1
    if judged and passed < judged:
        t4_warn = "%d/%d per-year judges below quality bar" % (judged - passed, judged)
    checks.append(_check("T4", warn_detail=t4_warn))
    if t4_warn:
        warnings.append("T4: " + t4_warn)

    # T5 — per-year gate blockers
    t5_warn = ""
    gate_blockers = 0
    for rec in records:
        if isinstance(rec, dict):
            yg = rec.get("year_gates")
        else:
            yg = rec.year_gates
        if not yg:
            continue
        if isinstance(yg, dict):
            gate_blockers += len(yg.get("blockers") or [])
        else:
            gate_blockers += len(yg.blockers or [])
    if gate_blockers:
        t5_warn = "%d per-year gate blocker(s) recorded" % gate_blockers
    checks.append(_check("T5", warn_detail=t5_warn))
    if t5_warn:
        warnings.append("T5: " + t5_warn)

    return GateReport(
        passed=len(blockers) == 0,
        blockers=blockers,
        warnings=warnings,
        checks=checks,
    )
