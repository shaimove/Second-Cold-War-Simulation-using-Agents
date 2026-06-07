"""Validation and repair pipeline for final orchestrator synthesis only.

Flow:
  1. Deterministic cleanup (fences, enums, string numbers, list coercion)
  2. Pydantic validate (`OrchestratorSynthesisOutput`)
  3. If still invalid → JSON Repair Agent (up to 2 attempts)
  4. If still invalid → rerun orchestrator synthesis once
  5. If still invalid → safe fallback dict (image generation disabled)
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import ValidationError

from .llm import LLMClient
from .schemas import (
    CONFIDENCE_LEVELS,
    DiscussionSummary,
    EVENT_STATUSES,
    IMPACT_LEVELS,
    OrchestratorSynthesisOutput,
    RunMetrics,
    YearBlock,
)
from .utils import extract_json, truncate

MAX_REPAIR_ATTEMPTS = 2

ORCHESTRATOR_SYNTHESIS_SCHEMA_DESCRIPTION = """
{
  "scenario_title": "string (required, short title, max 200 chars)",
  "scenario_summary": "string (required, 3-5 sentences, max 1200 chars)",
  "event_status": "one of: observed | hypothetical | mixed",
  "key_assumptions": ["string", "..."],
  "main_disagreements": ["string", "..."],
  "image_prompt": "string (non-graphic editorial illustration prompt, max 1200 chars)"
}
""".strip()

_ENUM_NORMALIZE = {
    "event_status": (EVENT_STATUSES, "hypothetical"),
    "impact": (IMPACT_LEVELS, "medium"),
    "confidence": (CONFIDENCE_LEVELS, "medium"),
    "domain": (
        ("economy", "strategy", "technology", "security", "ideology", "historical"),
        "strategy",
    ),
}


def format_validation_errors(exc: ValidationError) -> List[str]:
    return [
        "{loc}: {msg}".format(
            loc=".".join(str(x) for x in err.get("loc", ())),
            msg=err.get("msg", ""),
        )
        for err in exc.errors()
    ]


def coerce_probability(value: Any) -> float:
    """Convert string/int/float probabilities to clamped float."""
    if value is None:
        return 0.5
    if isinstance(value, (int, float)):
        v = float(value)
    elif isinstance(value, str):
        s = value.strip().replace(",", ".")
        if s.endswith("%"):
            try:
                v = float(s[:-1].strip()) / 100.0
            except ValueError:
                return 0.5
        else:
            try:
                v = float(s)
            except ValueError:
                return 0.5
    else:
        return 0.5
    if v > 1.0 and v <= 100.0:
        v = v / 100.0
    if v < 0:
        return 0.0
    if v > 1:
        return 1.0
    return v


def normalize_enum_value(
    field_name: str, value: Any, allowed: Tuple[str, ...], default: str
) -> str:
    if value is None:
        return default
    s = str(value).strip().lower()
    if s in allowed:
        return s
    # fuzzy: HIGH -> high already lowercased; hypothethical typo -> default
    for a in allowed:
        if a in s or s in a:
            return a
    return default


def _coerce_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip()[:400])
        return out
    return []


def _cleanup_timeline_events(obj: Any) -> Any:
    """Optional nested timeline with string probabilities / uppercase enums."""
    if not isinstance(obj, list):
        return obj
    cleaned = []
    for year_block in obj:
        if not isinstance(year_block, dict):
            continue
        yb = dict(year_block)
        if "year" in yb:
            try:
                yb["year"] = int(yb["year"])
            except (TypeError, ValueError):
                continue
        events = yb.get("events")
        if isinstance(events, list):
            new_events = []
            for ev in events:
                if not isinstance(ev, dict):
                    continue
                e = dict(ev)
                if "probability" in e:
                    e["probability"] = coerce_probability(e.get("probability"))
                if "impact" in e:
                    e["impact"] = normalize_enum_value(
                        "impact", e.get("impact"), IMPACT_LEVELS, "medium"
                    )
                if "confidence" in e:
                    e["confidence"] = normalize_enum_value(
                        "confidence", e.get("confidence"), CONFIDENCE_LEVELS, "medium"
                    )
                if "domain" in e:
                    e["domain"] = normalize_enum_value(
                        "domain",
                        e.get("domain"),
                        ("economy", "strategy", "technology", "security", "ideology", "historical"),
                        "strategy",
                    )
                new_events.append(e)
            yb["events"] = new_events
        cleaned.append(yb)
    return cleaned


def deterministic_cleanup(data: Any) -> Dict[str, Any]:
    """Best-effort normalization before Pydantic validation."""
    if data is None:
        return {}

    if isinstance(data, str):
        parsed = extract_json(data)
        if parsed is None:
            try:
                parsed = json.loads(data.strip())
            except Exception:
                return {}
        data = parsed

    if not isinstance(data, dict):
        return {}

    out: Dict[str, Any] = dict(data)

    for key in ("scenario_title", "scenario_summary", "image_prompt"):
        if key in out and out[key] is not None:
            out[key] = str(out[key]).strip()

    if "event_status" in out:
        out["event_status"] = normalize_enum_value(
            "event_status",
            out.get("event_status"),
            EVENT_STATUSES,
            "hypothetical",
        )

    for key in ("key_assumptions", "main_disagreements"):
        if key in out:
            out[key] = _coerce_str_list(out.get(key))

    if "timeline" in out:
        out["timeline"] = _cleanup_timeline_events(out.get("timeline"))

    # Strip unknown top-level keys that Pydantic would reject if we used extra=forbid
    allowed = {
        "scenario_title",
        "scenario_summary",
        "event_status",
        "key_assumptions",
        "main_disagreements",
        "image_prompt",
    }
    return {k: v for k, v in out.items() if k in allowed}


def validate_orchestrator_synthesis(
    data: Any,
) -> Tuple[Optional[OrchestratorSynthesisOutput], List[str]]:
    """Run cleanup then Pydantic validation. Never raises."""
    cleaned = deterministic_cleanup(data)
    try:
        model = OrchestratorSynthesisOutput.model_validate(cleaned)
        return model, []
    except ValidationError as e:
        return None, format_validation_errors(e)


def run_json_repair_agent(
    llm: LLMClient,
    raw_invalid: Any,
    validation_errors: List[str],
    seed: str,
    scenario_mode: str,
    discussion_summary: Optional[DiscussionSummary],
    attempt: int = 1,
) -> Dict[str, Any]:
    """LLM repair pass — JSON only, minimal semantic change."""
    if isinstance(raw_invalid, dict):
        raw_text = json.dumps(raw_invalid, ensure_ascii=False, default=str)
    else:
        raw_text = str(raw_invalid)

    summary_txt = (
        json.dumps(discussion_summary.model_dump(), ensure_ascii=False)
        if discussion_summary is not None
        else "<none>"
    )

    system = (
        "You are the JSON Repair Agent for the final orchestrator output. "
        "Fix ONLY structural/schema issues in the invalid JSON. "
        "Do NOT change the scenario meaning unless required to satisfy the schema. "
        "Return ONLY a single valid JSON object. No markdown."
    )

    user = (
        "Seed: " + seed + "\n"
        "Scenario mode: " + scenario_mode + "\n"
        "Repair attempt: " + str(attempt) + "\n\n"
        "Target schema:\n" + ORCHESTRATOR_SYNTHESIS_SCHEMA_DESCRIPTION + "\n\n"
        "Pydantic validation errors:\n"
        + "\n".join(validation_errors[:20])
        + "\n\n"
        "Compact discussion summary:\n"
        + truncate(summary_txt, 2000)
        + "\n\n"
        "Invalid raw output:\n"
        + truncate(raw_text, 4000)
        + "\n\n"
        "Return corrected JSON matching the target schema exactly."
    )

    return llm.call_llm_json(
        system_prompt=system,
        user_prompt=user,
        agent_name="orchestrator_json_repair",
        round_number=1000 + attempt,
        schema_name="json_repair",
        cache_context={
            "seed": seed,
            "mode": scenario_mode,
            "attempt": attempt,
            "errors": validation_errors[:5],
        },
        fallback={},
    )


def build_safe_fallback_synthesis(
    seed: str,
    scenario_mode: str,
    error_message: str,
    partial_timeline: Optional[List[YearBlock]] = None,
    last_summary: Optional[DiscussionSummary] = None,
) -> Dict[str, Any]:
    """Safe dict when validation, repair, and regeneration all fail."""
    disagreements: List[str] = []
    if last_summary is not None:
        disagreements = list(last_summary.areas_of_disagreement[:5])

    return {
        "scenario_title": "Scenario synthesis unavailable (validation failed)",
        "scenario_summary": (
            "The orchestrator could not produce a valid structured synthesis for: "
            + truncate(seed, 200)
            + ". Error: "
            + truncate(error_message, 300)
        ),
        "event_status": "hypothetical",
        "key_assumptions": ["Structured synthesis validation failed."],
        "main_disagreements": disagreements,
        "image_prompt": "",
        "error": error_message,
        "seed": seed,
        "scenario_mode": scenario_mode,
        "partial_timeline": [
            yb.model_dump() if hasattr(yb, "model_dump") else yb
            for yb in (partial_timeline or [])
        ],
        "image_generation_disabled": True,
    }


def synthesis_output_to_dict(
    model: OrchestratorSynthesisOutput,
) -> Dict[str, Any]:
    return model.model_dump()


def resolve_orchestrator_synthesis(
    llm: LLMClient,
    raw_output: Any,
    seed: str,
    scenario_mode: str,
    discussion_summary: Optional[DiscussionSummary],
    regenerate_fn: Callable[[], Any],
    partial_timeline: Optional[List[YearBlock]],
    metrics: RunMetrics,
) -> Tuple[Dict[str, Any], bool, bool]:
    """Validate/repair/regenerate/fallback pipeline.

    Returns:
        (synthesis_dict, validation_passed, image_generation_disabled)
    """
    metrics.synthesis_validation_errors = []
    metrics.synthesis_repair_attempts = 0
    metrics.synthesis_regeneration_attempts = 0
    metrics.synthesis_used_fallback = False
    metrics.synthesis_error_message = None

    def _try_validate(source: Any, label: str) -> Optional[OrchestratorSynthesisOutput]:
        model, errs = validate_orchestrator_synthesis(source)
        if model is not None:
            metrics.synthesis_validation_passed = True
            return model
        metrics.synthesis_validation_errors.extend(
            ["[{label}] {e}".format(label=label, e=e) for e in errs]
        )
        return None

    # --- Pass 1: initial output + cleanup ---
    model = _try_validate(raw_output, "initial")
    if model is not None:
        return synthesis_output_to_dict(model), True, False

    current_raw = raw_output

    # --- Pass 2: JSON repair agent (up to MAX_REPAIR_ATTEMPTS) ---
    for attempt in range(1, MAX_REPAIR_ATTEMPTS + 1):
        metrics.synthesis_repair_attempts = attempt
        errs = list(metrics.synthesis_validation_errors)
        repaired_raw = run_json_repair_agent(
            llm,
            current_raw,
            errs,
            seed,
            scenario_mode,
            discussion_summary,
            attempt=attempt,
        )
        current_raw = repaired_raw
        model = _try_validate(repaired_raw, "repair_{n}".format(n=attempt))
        if model is not None:
            return synthesis_output_to_dict(model), True, False

    # --- Pass 3: rerun final orchestrator synthesis once ---
    metrics.synthesis_regeneration_attempts = 1
    regenerated_raw = regenerate_fn()
    current_raw = regenerated_raw
    model = _try_validate(regenerated_raw, "regeneration")
    if model is not None:
        return synthesis_output_to_dict(model), True, False

    # Repair regenerated output once more (single extra repair cycle)
    for attempt in range(1, MAX_REPAIR_ATTEMPTS + 1):
        metrics.synthesis_repair_attempts = MAX_REPAIR_ATTEMPTS + attempt
        errs = list(metrics.synthesis_validation_errors)
        repaired_raw = run_json_repair_agent(
            llm,
            current_raw,
            errs,
            seed,
            scenario_mode,
            discussion_summary,
            attempt=MAX_REPAIR_ATTEMPTS + attempt,
        )
        current_raw = repaired_raw
        model = _try_validate(repaired_raw, "post_regen_repair_{n}".format(n=attempt))
        if model is not None:
            return synthesis_output_to_dict(model), True, False

    # --- Pass 4: safe fallback ---
    err_msg = (
        "Final orchestrator output failed validation after "
        + str(metrics.synthesis_repair_attempts)
        + " repair attempt(s) and "
        + str(metrics.synthesis_regeneration_attempts)
        + " regeneration(s)."
    )
    metrics.synthesis_used_fallback = True
    metrics.synthesis_error_message = err_msg
    metrics.synthesis_validation_passed = False
    fallback = build_safe_fallback_synthesis(
        seed=seed,
        scenario_mode=scenario_mode,
        error_message=err_msg,
        partial_timeline=partial_timeline,
        last_summary=discussion_summary,
    )
    return fallback, False, True
