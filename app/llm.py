"""LLM wrapper.

Provides:
- `call_llm_text` / `call_llm_json` used by every agent
- Deterministic mock mode when no API key is configured
- Optional SQLite cache keyed by (model + agent + context hash)
- Per-run metrics (LLM call count, cache hits, est. tokens)
- Simple retry with backoff

The OpenAI SDK is imported lazily so the module is importable even when
the package isn't installed (useful for `pytest` in mock mode).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from . import config as _config_mod
from . import db
from .utils import estimate_tokens, extract_json, stable_hash, truncate


# Agents that use OPENAI_ORCHESTRATOR_MODEL (stronger model for synthesis).
ORCHESTRATOR_AGENT_NAMES = frozenset(
    {
        "orchestrator_summary",
        "orchestrator_final",
        "orchestrator_json_repair",
    }
)

JUDGE_AGENT_NAMES = frozenset(
    {"quality_judge", "year_quality_judge", "timeline_quality_judge"}
)


@dataclass
class LLMMetrics:
    llm_calls: int = 0
    cache_hits: int = 0
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    agents_used: list = field(default_factory=list)

    def record_call(self, agent_name: str, prompt_chars: int, output_chars: int) -> None:
        self.llm_calls += 1
        if agent_name not in self.agents_used:
            self.agents_used.append(agent_name)
        self.estimated_input_tokens += estimate_tokens(" " * prompt_chars)
        self.estimated_output_tokens += estimate_tokens(" " * output_chars)

    def record_cache_hit(self, agent_name: str) -> None:
        self.cache_hits += 1
        if agent_name not in self.agents_used:
            self.agents_used.append(agent_name)


class LLMClient:
    """Thin wrapper around the OpenAI SDK with mock + cache support."""

    def __init__(self, config=None, metrics: Optional[LLMMetrics] = None):
        self.config = config or _config_mod.CONFIG
        self.metrics = metrics or LLMMetrics()
        self._client = None

    # -- Public API ---------------------------------------------------------

    def call_llm_text(
        self,
        system_prompt: str,
        user_prompt: str,
        agent_name: str,
        round_number: int = 0,
        cache_context: Optional[Dict[str, Any]] = None,
        temperature: float = 0.4,
    ) -> str:
        cache_key = self._cache_key(
            agent_name, round_number, system_prompt, user_prompt, cache_context
        )
        if self.config.use_llm_cache:
            cached = db.cache_get(cache_key)
            if cached and "text" in cached:
                self.metrics.record_cache_hit(agent_name)
                return cached["text"]

        if self.config.mock_mode:
            text = _mock_text(agent_name, user_prompt)
        else:
            text = self._call_openai_text(
                system_prompt,
                user_prompt,
                temperature,
                model=self._resolve_model(agent_name),
            )

        self.metrics.record_call(
            agent_name,
            prompt_chars=len(system_prompt) + len(user_prompt),
            output_chars=len(text or ""),
        )

        if self.config.use_llm_cache:
            db.cache_set(
                cache_key,
                self._resolve_model(agent_name),
                agent_name,
                {"text": text},
            )
        return text

    def call_llm_json(
        self,
        system_prompt: str,
        user_prompt: str,
        agent_name: str,
        round_number: int = 0,
        cache_context: Optional[Dict[str, Any]] = None,
        schema_name: Optional[str] = None,
        temperature: float = 0.3,
        fallback: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cache_key = self._cache_key(
            agent_name, round_number, system_prompt, user_prompt, cache_context
        )
        if self.config.use_llm_cache:
            cached = db.cache_get(cache_key)
            if cached and "json" in cached:
                self.metrics.record_cache_hit(agent_name)
                return cached["json"]

        if self.config.mock_mode:
            data = _mock_json(agent_name, user_prompt, round_number, schema_name)
            text_for_metrics = json.dumps(data)
        else:
            raw = self._call_openai_text(
                system_prompt + _JSON_INSTRUCTION,
                user_prompt,
                temperature,
                force_json=True,
                model=self._resolve_model(agent_name),
            )
            data = extract_json(raw)
            if data is None:
                data = fallback or {"error": "invalid_json", "raw": truncate(raw, 1000)}
            text_for_metrics = raw

        self.metrics.record_call(
            agent_name,
            prompt_chars=len(system_prompt) + len(user_prompt),
            output_chars=len(text_for_metrics or ""),
        )

        if self.config.use_llm_cache:
            db.cache_set(
                cache_key,
                self._resolve_model(agent_name),
                agent_name,
                {"json": data},
            )
        return data

    # -- Internals ----------------------------------------------------------

    def _resolve_model(self, agent_name: str) -> str:
        if agent_name in JUDGE_AGENT_NAMES:
            return self.config.openai_judge_model
        if agent_name in ORCHESTRATOR_AGENT_NAMES:
            return self.config.openai_orchestrator_model
        return self.config.openai_model

    def _cache_key(
        self,
        agent_name: str,
        round_number: int,
        system_prompt: str,
        user_prompt: str,
        cache_context: Optional[Dict[str, Any]],
    ) -> str:
        return stable_hash(
            self._resolve_model(agent_name),
            agent_name,
            round_number,
            system_prompt,
            user_prompt,
            cache_context or {},
        )

    def _get_client(self):
        if self._client is not None:
            return self._client
        from openai import OpenAI  # imported lazily

        self._client = OpenAI(api_key=self.config.openai_api_key)
        return self._client

    def _call_openai_text(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        force_json: bool = False,
        max_retries: int = 2,
        model: Optional[str] = None,
    ) -> str:
        model = model or self.config.openai_model
        last_err: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                client = self._get_client()
                kwargs: Dict[str, Any] = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": temperature,
                }
                if force_json:
                    kwargs["response_format"] = {"type": "json_object"}
                resp = client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except Exception as e:  # broad: SDK raises many error subclasses
                last_err = e
                if attempt < max_retries:
                    time.sleep(0.8 * (attempt + 1))
                else:
                    break
        raise RuntimeError(
            "OpenAI request failed after retries: " + str(last_err)
        )


_JSON_INSTRUCTION = (
    "\n\nReturn ONLY a single valid JSON object. "
    "No markdown, no commentary, no code fences."
)


# --- Mock responses --------------------------------------------------------
#
# When OPENAI_API_KEY is not set, we synthesize structured outputs so the
# graph, API, and tests can run without any network access. These mocks
# are intentionally schema-faithful but content-light.

_MOCK_DOMAINS = {
    "geo_strategy": "strategy",
    "economy_technology": "economy",
    "domestic_ideology": "ideology",
    "security_taiwan": "security",
    "historical_analogy": "historical",
}


def _mock_text(agent_name: str, user_prompt: str) -> str:
    snippet = truncate(user_prompt.replace("\n", " "), 120)
    return (
        "[MOCK:{agent}] Plausible-but-fictional analytic note. "
        "Context excerpt: {ctx}".format(agent=agent_name, ctx=snippet)
    )


def _mock_json(
    agent_name: str,
    user_prompt: str,
    round_number: int,
    schema_name: Optional[str],
) -> Dict[str, Any]:
    if schema_name == "evidence_summary" or agent_name == "evidence_rag":
        return {
            "observed_facts": [
                "U.S.-China technology competition is a long-running structural trend."
            ],
            "historical_analogies": [
                "U.S.-USSR rivalry shows how blocs form under sustained competition."
            ],
            "strategy_frameworks": [
                "Deterrence, economic statecraft, and alliance management."
            ],
            "hypothetical_assumptions": [
                "The user's seed describes a future-state assumption, not a fact."
            ],
            "sources": [],
            "compact_summary": (
                "Mock evidence: structural rivalry shaped by tech, trade, "
                "Taiwan and alliance dynamics; user seed treated as hypothetical."
            ),
            "note": "Mock mode active - no real retrieval performed.",
        }

    if schema_name == "discussion_summary" or agent_name == "orchestrator_summary":
        year = 2026
        if "Target year:" in user_prompt:
            for part in user_prompt.split("Target year:")[1].split("\n"):
                digits = "".join(ch for ch in part.strip() if ch.isdigit())
                if digits:
                    year = int(digits[:4])
                    break
        return {
            "round_number": round_number or 1,
            "target_year": year,
            "areas_of_agreement": [
                "Tech and trade decoupling continues unevenly.",
                "Taiwan remains the central flashpoint.",
            ],
            "areas_of_disagreement": [
                "Pace of decoupling vs. selective re-engagement.",
            ],
            "emerging_timeline": [
                "{y}: heightened tech controls".format(y=year),
            ],
            "key_uncertainties": [
                "Domestic political shocks in either country.",
            ],
            "agent_positions": {
                "geo_strategy": "Alliances tighten in the Indo-Pacific.",
                "economy_technology": "Tech bifurcation accelerates.",
                "domestic_ideology": "Nationalism remains an amplifier.",
                "security_taiwan": "Gray-zone pressure rises, full conflict still unlikely.",
                "historical_analogy": "Useful but imperfect Cold War parallels.",
            },
        }

    if schema_name == "year_decision" or agent_name == "orchestrator_year_decision":
        year = round_number or 2026
        return {
            "year": year,
            "headline": "Mock locked outcome for {y}".format(y=year),
            "events": [
                {
                    "event": "Mock strategic development in {y}".format(y=year),
                    "domain": "strategy",
                    "probability": 0.5,
                    "impact": "medium",
                    "confidence": "medium",
                    "rationale": "Mock year decision.",
                }
            ],
        }

    if schema_name == "red_team" or agent_name == "red_team":
        return {
            "agent_name": "red_team",
            "round_number": round_number or 1,
            "main_assessment": (
                "Scenario may overweight linear escalation; "
                "watch for surprise de-escalation or economic shocks reshaping incentives."
            ),
            "key_drivers": [
                "Overconfidence in alliance cohesion",
                "Underweighted economic interdependence",
            ],
            "timeline_contributions": [],
            "risks": [
                "Black-swan domestic crises ignored",
                "Cyber/space domains under-modeled",
            ],
            "uncertainties": ["Leadership succession risks"],
            "agreements": [],
            "disagreements": ["Pace of decoupling assumed too steady"],
            "position_changed_from_previous_round": False,
            "sources_used": [],
            "grounding_notes": [],
            "rag_influence": "not_used",
            "rag_influence_explanation": "",
            "findings": [
                {
                    "issue": "Assumes steady decoupling",
                    "severity": "medium",
                    "affected_assumption": "Linear tech bifurcation",
                },
                {
                    "issue": "Underweights economic interdependence",
                    "severity": "medium",
                    "affected_assumption": "Decoupling speed",
                },
            ],
        }

    if schema_name == "json_repair" or agent_name == "orchestrator_json_repair":
        return {
            "scenario_title": "Repaired USA-China Scenario (Mock)",
            "scenario_summary": "Mock repaired synthesis preserving seed meaning.",
            "event_status": "hypothetical",
            "key_assumptions": ["Repaired output"],
            "main_disagreements": ["Repair agent restored schema"],
            "image_prompt": _MOCK_IMAGE_PROMPT,
        }

    if schema_name == "year_judge_verdict" or agent_name == "year_quality_judge":
        year = round_number or 2026
        return {
            "overall_score": 4.0,
            "dimensions": [
                {"name": "year_scope", "score": 4, "rationale": "Mock: locked year only."},
                {"name": "seed_fidelity", "score": 4, "rationale": "Mock: seed reflected."},
                {
                    "name": "discussion_fidelity",
                    "score": 4,
                    "rationale": "Mock: reflects agent positions.",
                },
                {
                    "name": "prior_timeline_coherence",
                    "score": 4,
                    "rationale": "Mock: consistent with prior years.",
                },
                {
                    "name": "uncertainty_preservation",
                    "score": 3,
                    "rationale": "Mock: some uncertainty retained.",
                },
            ],
            "failure_modes": [],
            "one_line_verdict": "Mock year judge: acceptable lock for %d." % year,
            "summary_paragraph": (
                "The mock year judge found this year lock structurally acceptable: "
                "it stays on the target year, aligns with the seed, and reflects "
                "specialist discussion without erasing all uncertainty."
            ),
        }

    if schema_name == "timeline_judge_verdict" or agent_name == "timeline_quality_judge":
        return {
            "overall_score": 4.0,
            "dimensions": [
                {
                    "name": "timeline_completeness",
                    "score": 4,
                    "rationale": "Mock: six years locked with headlines.",
                },
                {"name": "seed_fidelity", "score": 4, "rationale": "Mock: arc reflects seed."},
                {
                    "name": "cross_year_coherence",
                    "score": 4,
                    "rationale": "Mock: years connect logically.",
                },
                {
                    "name": "escalation_arc",
                    "score": 3,
                    "rationale": "Mock: moderate escalation path.",
                },
                {
                    "name": "uncertainty_preservation",
                    "score": 3,
                    "rationale": "Mock: some open questions remain.",
                },
            ],
            "failure_modes": [],
            "one_line_verdict": "Mock timeline judge: coherent six-year arc.",
            "summary_paragraph": (
                "The mock timeline judge found the locked 2026-2031 arc structurally "
                "complete and seed-aligned, with a plausible cross-year escalation "
                "path and retained uncertainties suitable for red-team critique."
            ),
        }

    if schema_name == "judge_verdict" or agent_name == "quality_judge":
        return {
            "overall_score": 4.0,
            "dimensions": [
                {
                    "name": "seed_fidelity",
                    "score": 4,
                    "rationale": "Mock: seed reflected in summary.",
                },
                {
                    "name": "plausibility",
                    "score": 4,
                    "rationale": "Mock: framed as plausible scenario.",
                },
                {
                    "name": "specialist_diversity",
                    "score": 3,
                    "rationale": "Mock: agents moderately distinct.",
                },
                {
                    "name": "disagreement_preservation",
                    "score": 4,
                    "rationale": "Mock: disagreements listed.",
                },
                {
                    "name": "timeline_usefulness",
                    "score": 4,
                    "rationale": "Mock: timeline spans 2026-2031.",
                },
            ],
            "failure_modes": [],
            "one_line_verdict": "Mock judge: acceptable quality for portfolio demo.",
            "summary_paragraph": (
                "This mock scenario meets basic structural expectations: the seed is "
                "reflected in the title and summary, specialist agents offer distinct "
                "angles, and disagreements are preserved rather than collapsed. The "
                "timeline spans 2026–2031 with usable headlines. For a portfolio demo "
                "the output is strategically readable, though a live judge would score "
                "nuance and grounding more precisely."
            ),
        }

    if schema_name == "final_synthesis" or agent_name == "orchestrator_final":
        return {
            "scenario_title": "One Plausible USA-China Rivalry Path, 2026-2031 (Mock)",
            "scenario_summary": (
                "A mock five-year scenario where tech controls tighten, alliances "
                "recalibrate, and Taiwan remains the central but managed flashpoint."
            ),
            "event_status": "hypothetical",
            "key_assumptions": [
                "No major hot war between great powers.",
                "Continued semiconductor competition.",
            ],
            "main_disagreements": [
                "Pace of economic decoupling.",
            ],
            "image_prompt": _MOCK_IMAGE_PROMPT,
        }

    domain = _MOCK_DOMAINS.get(agent_name, "strategy")
    year = 2026
    if "Target simulation year:" in user_prompt:
        for part in user_prompt.split("Target simulation year:")[1].split("\n"):
            digits = "".join(ch for ch in part.strip() if ch.isdigit())
            if digits:
                year = int(digits[:4])
                break
    return {
        "agent_name": agent_name,
        "round_number": round_number or 1,
        "main_assessment": (
            "[MOCK:{a}] Strategic-level assessment for {y}, "
            "without operational detail."
        ).format(a=agent_name, y=year),
        "key_drivers": [
            "Structural rivalry",
            "Tech competition",
            "Alliance dynamics",
        ],
        "timeline_contributions": [
            {
                "year": year,
                "event": "Mock development shaping the rivalry in {y}.".format(y=year),
                "probability": 0.6,
                "impact": "medium",
                "confidence": "medium",
                "rationale": "Mock single-year contribution.",
            },
        ],
        "risks": ["Misperception", "Accidental escalation"],
        "uncertainties": ["Domestic political shocks"],
        "agreements": ["Taiwan is central"],
        "disagreements": ["Pace of decoupling"],
        "position_changed_from_previous_round": round_number > 1,
        "sources_used": [],
        "grounding_notes": [],
        "rag_influence": "not_used",
        "rag_influence_explanation": "",
        "_mock_domain": domain,
    }


_MOCK_IMAGE_PROMPT = (
    "A cinematic geopolitical editorial illustration of the Pacific region, "
    "Washington, Beijing, and Taipei connected by glowing trade routes, "
    "semiconductor circuits, diplomatic chess pieces, non-violent naval "
    "silhouettes in the background, divided technology networks, "
    "serious dark-blue tone, high detail, no graphic content."
)
