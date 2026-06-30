"""Small reusable helpers (hashing, JSON parsing, IDs, truncation)."""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


SIMULATION_YEARS = [2026, 2027, 2028, 2029, 2030, 2031]


def new_run_id() -> str:
    return "run_" + uuid.uuid4().hex[:12]


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hash(*parts: Any) -> str:
    """Deterministic hash for cache keys."""
    h = hashlib.sha256()
    for part in parts:
        h.update(json.dumps(part, sort_keys=True, default=str).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()


def truncate(text: str, max_chars: int) -> str:
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def assemble_user_prompt(context: str, fixed_suffix: str, max_chars: int) -> str:
    """Build a user prompt that never truncates the fixed suffix (schema, citations).

    Trims `context` first so JSON instructions stay intact when the budget is tight.
    """
    fixed_suffix = fixed_suffix or ""
    context = context or ""
    if len(fixed_suffix) >= max_chars:
        return truncate(fixed_suffix, max_chars)
    budget = max_chars - len(fixed_suffix)
    return budget_prompt_sections([(context, 1)], budget) + fixed_suffix


def budget_prompt_sections(
    sections: List[Tuple[str, int]],
    max_chars: int,
) -> str:
    """Join prompt sections; trim higher `trim_priority` sections first when over budget.

    Lower priority numbers are preserved longer (e.g. 1 = seed header, 5 = verbose blob).
    """
    if max_chars <= 0:
        return ""
    if not sections:
        return ""

    parts: List[List] = [[text or "", priority] for text, priority in sections]

    def joined() -> str:
        return "\n\n".join(p[0] for p in parts if p[0])

    if len(joined()) <= max_chars:
        return joined()

    while len(joined()) > max_chars:
        trim_priority = max(p[1] for p in parts)
        candidates = [p for p in parts if p[1] == trim_priority and len(p[0]) > 40]
        if not candidates:
            return truncate(joined(), max_chars)
        target = candidates[0]
        overflow = len(joined()) - max_chars
        new_len = max(40, len(target[0]) - max(overflow, 80))
        target[0] = truncate(target[0], new_len)

    return joined()


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON parse.

    Tolerates code fences and surrounding prose. Returns None if no JSON
    object can be recovered.
    """
    if text is None:
        return None
    candidates = []
    fenced = _FENCE_RE.findall(text)
    candidates.extend(fenced)
    candidates.append(text)
    for cand in candidates:
        cand = cand.strip()
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        start = cand.find("{")
        end = cand.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = cand[start : end + 1]
            try:
                obj = json.loads(snippet)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    return None


def estimate_tokens(text: str) -> int:
    """Cheap heuristic: ~4 chars per token. Used only for metrics."""
    if not text:
        return 0
    return max(1, len(text) // 4)
