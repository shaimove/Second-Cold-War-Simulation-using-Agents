"""Orchestrate gates + judge and persist monitor block on saved runs."""
from __future__ import annotations

from typing import Any, Dict, Optional, Union

from .. import config as _config_mod
from .. import db
from ..llm import LLMClient
from ..schemas import FinalScenario, MonitorResult
from ..utils import utcnow_iso
from .gates import check_hard_gates
from .judge import run_quality_judge


def run_monitor(
    final: Union[FinalScenario, Dict[str, Any]],
    llm: Optional[LLMClient] = None,
    *,
    skip_judge: bool = False,
    force_judge: bool = False,
) -> MonitorResult:
    """Run hard gates; run LLM judge unless blockers, skip_judge, or force_judge."""
    if isinstance(final, FinalScenario):
        data = final.model_dump()
    else:
        data = dict(final)

    gates = check_hard_gates(data)
    cfg = _config_mod.CONFIG
    judge_model = getattr(cfg, "openai_judge_model", cfg.openai_orchestrator_model)

    result = MonitorResult(
        gates=gates,
        judged_at=utcnow_iso(),
        judge_model=judge_model,
    )

    if skip_judge:
        result.judge_skipped = True
        result.judge_skip_reason = "Gates only (judge not requested)."
        return result

    if gates.blockers and not force_judge:
        result.judge_skipped = True
        result.judge_skip_reason = "Hard gate blockers: " + "; ".join(gates.blockers)
        return result

    try:
        verdict = run_quality_judge(data, llm=llm)
        result.judge = verdict
    except Exception as e:
        result.judge_skipped = True
        result.judge_error = str(e)[:400]
        result.judge_skip_reason = "Judge call failed"

    return result


def attach_monitor(payload: Dict[str, Any], monitor: MonitorResult) -> Dict[str, Any]:
    out = dict(payload)
    out["monitor"] = monitor.model_dump()
    return out


def attach_gates_to_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Free deterministic gates after every simulation (no LLM)."""
    return attach_monitor(payload, run_monitor(payload, skip_judge=True))


def finalize_run_payload(payload: Dict[str, Any], llm: Optional[LLMClient] = None) -> Dict[str, Any]:
    """Attach monitor (gates + optional judge) and persist to SQLite."""
    cfg = _config_mod.CONFIG
    if cfg.enable_run_judge:
        monitor = run_monitor(payload, llm=llm)
    else:
        monitor = run_monitor(payload, skip_judge=True)
    updated = attach_monitor(payload, monitor)
    db.save_scenario_run(
        run_id=updated.get("run_id") or "",
        seed=updated.get("seed") or "",
        scenario_mode=updated.get("scenario_mode") or "base_case",
        scenario_title=updated.get("scenario_title") or "",
        full_json=updated,
    )
    return updated


def judge_saved_run(
    run_id: str,
    llm: Optional[LLMClient] = None,
    *,
    force: bool = False,
) -> Optional[Dict[str, Any]]:
    """Load run from SQLite, judge, save back. Returns updated full_json."""
    payload = db.load_scenario_run(run_id)
    if payload is None:
        return None

    monitor = run_monitor(payload, llm=llm, force_judge=force)
    updated = attach_monitor(payload, monitor)

    db.save_scenario_run(
        run_id=updated.get("run_id") or run_id,
        seed=updated.get("seed") or "",
        scenario_mode=updated.get("scenario_mode") or "base_case",
        scenario_title=updated.get("scenario_title") or "",
        full_json=updated,
    )
    return updated
