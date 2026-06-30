# Prompt development guide

All LLM **system prompts** and reusable prompt fragments live in one place:

**[`app/prompts.py`](../app/prompts.py)**

Edit that file to tune agent behavior. Runtime code in [`app/agents.py`](../app/agents.py) builds **user prompts** from scenario state (seed, year, evidence, prior timeline) and calls the LLM through [`app/llm.py`](../app/llm.py).

## What is in `app/prompts.py`

| Symbol | Used by |
|--------|---------|
| `AGENT_SYSTEM_PROMPTS` | Five domain agents (geo, economy, domestic, security, historical) |
| `DOMAIN_AGENT_YEAR_FOCUS` | Appended to every domain agent system prompt |
| `domain_agent_schema_hint(year)` | JSON output contract for domain agents |
| `EVIDENCE_AGENT_SYSTEM` | Evidence / RAG agent |
| `RED_TEAM_SYSTEM` | Red-team agent |
| `ORCHESTRATOR_SUMMARY_SYSTEM` | Per-round compression within a year |
| `ORCHESTRATOR_YEAR_DECISION_SYSTEM` | Locks one simulation year after discussion |
| `ORCHESTRATOR_FINAL_SYSTEM` | Final title/summary/image (timeline already locked) |
| `SAFETY_TAIL` | Appended to most agents |
| `JSON_TAIL` | “Return JSON only” instruction |

## User prompt assembly (not in `prompts.py`)

These are built in code when a run executes:

| Agent | Builder |
|-------|---------|
| Evidence | `run_evidence_agent()` in `app/agents.py` |
| Domain agents | `run_domain_agent()` — includes target year + locked prior years |
| Orchestrator summary | `run_orchestrator_summary()` |
| Orchestrator year decision | `run_orchestrator_year_decision()` |
| Red team | `run_red_team_agent()` |
| Final synthesis | `run_orchestrator_final_synthesis_raw()` |

To change **what context agents see** (e.g. more prior-year history), edit those functions in `app/agents.py` or add helpers like `format_resolved_timeline()` there.

## Other prompt locations

| File | Purpose |
|------|---------|
| `app/rag_citations.py` | Citation appendix appended to domain/red-team JSON prompts |
| `app/final_output_validation.py` | JSON repair prompt for broken final synthesis |
| `app/monitor/judge.py` | Quality judge rubric |
| `app/image_generation.py` | Image prompt wrapper (safety tail) |
| `app/llm.py` | Mock JSON stubs when `OPENAI_API_KEY` is unset (tests/CI) |

## Per-year pipeline (current behavior)

1. **Evidence** — once at start  
2. For each year **2026 → 2031**: up to **3 discussion iterations** among domain agents, then **orchestrator year decision** (locked timeline)  
3. **Red team** — once on full timeline  
4. **Final orchestrator** — narrative + image from locked timeline  

Domain agents only discuss the **current target year**; earlier years are passed as **locked history**.
