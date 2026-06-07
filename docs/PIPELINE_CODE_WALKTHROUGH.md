# Code Pipeline Walkthrough (file + line + function)

This document maps **every major step** of a simulation run to the exact **file**, **line range**, and **function** that executes it, plus what logic happens there.

For a higher-level overview (diagrams, concepts), see [PIPELINE.md](PIPELINE.md).

> Line numbers refer to the current codebase on branch `main`. If you edit files, re-check line numbers.

---

## Table of contents

1. [Startup and configuration](#1-startup-and-configuration)
2. [Frontend: user clicks Run Simulation](#2-frontend-user-clicks-run-simulation)
3. [API entry: POST /api/run-scenario](#3-api-entry-post-apirun-scenario)
4. [Graph runner: run_graph()](#4-graph-runner-run_graph)
5. [Node ① orchestrator_initialize](#5-node--orchestrator_initialize)
6. [Node ② evidence_rag_agent](#6-node--evidence_rag_agent)
7. [Nodes ③–⑧ Discussion rounds + summaries](#7-nodes-38-discussion-rounds--summaries)
8. [Node ⑨ red_team_agent](#8-node--red_team_agent)
9. [Node ⑩ orchestrator_synthesis](#9-node--orchestrator_synthesis)
10. [Node ⑪ orchestrator_image_generation](#10-node--orchestrator_image_generation)
11. [Node ⑫ save_run](#11-node--save_run)
12. [Post-graph: metrics + second save + return](#12-post-graph-metrics--second-save--return)
13. [Shared subsystem: LLMClient.call_llm_json](#13-shared-subsystem-llmclientcall_llm_json)
14. [Shared subsystem: RAG retrieval](#14-shared-subsystem-rag-retrieval)
15. [Shared subsystem: cost control helpers](#15-shared-subsystem-cost-control-helpers)
16. [Frontend: render result](#16-frontend-render-result)
17. [Quick reference: all LLM calls per run](#17-quick-reference-all-llm-calls-per-run)

---

## 1. Startup and configuration

| Step | File | Lines | Function | Logic |
|---|---|---|---|---|
| Load `.env` | `app/config.py` | 14–16 | `load_dotenv()` | Reads environment variables from `.env` at import time |
| Build config object | `app/config.py` | 61–79 | `load_config()` | Parses bools/ints, paths, model names into a frozen `Config` dataclass |
| Global config singleton | `app/config.py` | 82 | `CONFIG = load_config()` | All modules read `_config_mod.CONFIG` |
| Mock mode check | `app/config.py` | 55–58 | `Config.mock_mode` property | Returns `True` when `OPENAI_API_KEY` is empty |
| Init SQLite tables | `app/db.py` | 61–65 | `init_db()` | Creates `scenario_runs` and `llm_cache` tables if missing |
| FastAPI lifespan hook | `app/main.py` | 28–31 | `_lifespan()` | Calls `db.init_db()` when uvicorn starts |

**Key env vars read here:** `OPENAI_API_KEY`, `OPENAI_MODEL`, `USE_RAG`, `USE_LLM_CACHE`, `MAX_AGENT_DISCUSSION_ROUNDS`, etc. (see `.env.example`).

---

## 2. Frontend: user clicks Run Simulation

| Step | File | Lines | Function | Logic |
|---|---|---|---|---|
| Page init | `frontend/app.js` | 296–302 | `init()` | Wires button click, loads config + saved runs |
| Button handler | `frontend/app.js` | 298 | `$("#run").addEventListener("click", runScenario)` | Binds **Run Simulation** |
| Validate seed | `frontend/app.js` | 242–247 | `runScenario()` | Returns early if textarea is empty |
| Disable UI + show status | `frontend/app.js` | 249–252 | `runScenario()` | Disables button, hides empty state |
| Animate progress (cosmetic) | `frontend/app.js` | 254–262 | `runScenario()` | Cycles through 9 progress steps every 700ms (backend is one blocking call) |
| **HTTP POST** | `frontend/app.js` | 265–268 | `runScenario()` | `fetch("/api/run-scenario", { seed, scenario_mode })` |
| Parse JSON response | `frontend/app.js` | 274 | `runScenario()` | `const data = await r.json()` |
| Render dashboard | `frontend/app.js` | 277 | `runScenario()` | Calls `renderResult(data)` |
| Refresh sidebar | `frontend/app.js` | 279 | `runScenario()` | Calls `loadSavedRuns()` |

**Request body shape** (validated by Pydantic in `app/schemas.py:190–201`):

```json
{ "seed": "...", "scenario_mode": "base_case" }
```

Valid modes: `base_case`, `escalation`, `de_escalation`, `wildcard` (`app/schemas.py:14`).

---

## 3. API entry: POST /api/run-scenario

| Step | File | Lines | Function | Logic |
|---|---|---|---|---|
| Route handler | `app/main.py` | 83–88 | `run_scenario(req)` | FastAPI endpoint |
| Reject empty seed | `app/main.py` | 85–86 | `run_scenario()` | Raises HTTP 400 if seed is blank |
| **Start pipeline** | `app/main.py` | 87 | `run_scenario()` | `final = run_graph(seed=..., scenario_mode=...)` |
| Serialize response | `app/main.py` | 88 | `run_scenario()` | `return final.model_dump()` → JSON to browser |

---

## 4. Graph runner: run_graph()

**File:** `app/graph.py`  
**Function:** `run_graph()` — lines **335–380**

| Line(s) | What happens |
|---|---|
| 341 | Creates `LLMClient()` — one client shared by all agents for this run |
| 342 | Creates `ScenarioState(run_id=new_run_id(), seed, scenario_mode)` — `new_run_id()` at `app/utils.py:15–16` |
| 343 | Initializes empty `RunMetrics()` |
| 344 | Starts elapsed-time timer |
| 346 | Calls `build_graph()` (lines 291–332) to compile LangGraph |
| 347–356 | **If LangGraph works:** `compiled.invoke(payload)` runs all 12 nodes |
| 357–358 | **Else:** `_run_sequential()` (lines 383–389) loops `NODES` in order |
| 360–365 | Copies LLM metrics into `state.run_metrics` (calls, cache hits, tokens, agents used) |
| 367 | `build_final_scenario(state)` (lines 232–269) → `FinalScenario` |
| 369–376 | Second DB save with final metrics |
| 380 | Returns `FinalScenario` to API |

### build_graph() — lines 291–332

| Line(s) | Logic |
|---|---|
| 298–301 | Tries `from langgraph.graph import StateGraph, END`; returns `None` on failure |
| 303 | Creates `StateGraph(dict)` |
| 306–321 | `make_wrapper(fn)` — converts dict state ↔ `ScenarioState`, passes `_llm` through |
| 324–330 | Adds each node from `NODES` list (lines 275–288) as sequential edges |
| 332 | Returns `graph.compile()` |

### _run_sequential() — lines 383–389

Fallback when LangGraph unavailable or throws:

```python
for _name, fn in NODES:
    state = fn(state, llm)   # same 12 functions, same order
```

Errors are appended to `state.errors` but do not stop the pipeline.

### NODES list — lines 275–288

Fixed order (no dynamic routing):

```
orchestrator_initialize
→ evidence_rag_agent
→ discussion_round_1 → orchestrator_summarize_round_1
→ discussion_round_2 → orchestrator_summarize_round_2
→ discussion_round_3 → orchestrator_summarize_round_3
→ red_team_agent
→ orchestrator_synthesis
→ orchestrator_image_generation
→ save_run
```

---

## 5. Node ① orchestrator_initialize

| | |
|---|---|
| **Graph node** | `app/graph.py:45–53` → `orchestrator_initialize()` |
| **Called from** | `NODES[0]` |

| Line | Logic |
|---|---|
| 46–47 | If `event_status` empty → `agent_mod.classify_event_status(state.seed)` |
| 48–49 | If no title → `"USA-China Scenario: " + truncate(seed, 80)` |
| 51–52 | For each name in `DOMAIN_AGENTS`, creates empty list in `state.agent_outputs` |

### classify_event_status() — `app/agents.py:485–503`

| Logic |
|---|
| Scans seed for future markers (`"will"`, `"if "`, `"suppose"`, …) and past markers |
| Returns `"hypothetical"`, `"mixed"`, or defaults to `"hypothetical"` |
| No LLM call — pure string heuristics |

**State after this node:** `run_id`, `seed`, `scenario_mode`, `event_status`, placeholder title, empty agent output slots.

---

## 6. Node ② evidence_rag_agent

| | |
|---|---|
| **Graph node** | `app/graph.py:56–64` → `evidence_rag_node()` |
| **Agent function** | `app/agents.py:108–185` → `run_evidence_agent()` |

### Step-by-step inside run_evidence_agent()

| Step | File:Lines | Function | Logic |
|---|---|---|---|
| RAG retrieve (if enabled) | `agents.py:116–117` | `retrieve_with_cache(seed, scenario_mode)` | Only if `USE_RAG=true` |
| Format chunks for prompt | `agents.py:119–127` | inline | Joins chunk text with source metadata, truncated to 400 chars each |
| Collect source paths | `agents.py:128` | inline | Sorted unique file paths for `EvidenceSummary.sources` |
| Build system prompt | `agents.py:130–137` | inline | Evidence agent role + `SAFETY_TAIL` + `JSON_TAIL` |
| Build user prompt | `agents.py:139–152` | inline | Seed, mode, retrieved text (or `<none>`), JSON schema |
| **LLM call** | `agents.py:154–173` | `llm.call_llm_json(...)` | `agent_name="evidence_rag"`, `schema_name="evidence_summary"` |
| Parse into model | `agents.py:175–184` | inline | Builds `EvidenceSummary` Pydantic object |
| Return | `agents.py:185` | inline | `(summary, len(chunks), cache_hit)` |

### Graph node writes back — `graph.py:60–63`

```
state.evidence_summary = summary
state.run_metrics.retrieved_docs = n_chunks
if cache_hit: state.run_metrics.cache_hits += 1
```

**Important:** This is the **only** step that reads RAG documents. All later agents get a compressed summary via `compact_evidence_for_agents()`.

---

## 7. Nodes ③–⑧ Discussion rounds + summaries

### 7a. Discussion round (runs 3×: rounds 1, 2, 3)

| | |
|---|---|
| **Core function** | `app/graph.py:67–101` → `_run_discussion_round(state, llm, round_number)` |
| **Wrappers** | `discussion_round_1` (104–105), `discussion_round_2` (107–111), `discussion_round_3` (114–119) |

| Line | Logic |
|---|---|
| 70–71 | If `_early_stopped` flag set → return immediately (skip round) |
| 73 | `compact_evidence_for_agents(state.evidence_summary)` → one short paragraph (`cost_control.py:21–44`) |
| 74–76 | Load previous round's `DiscussionSummary` from `state.discussion_rounds[-1]` (None in round 1) |
| 79–96 | **Loop over 5 domain agents** (sequential, not parallel): |
| 80–83 | Get agent's own prior output; compress with `compact_agent_position()` (`cost_control.py:47–54`) |
| 85–94 | Call `agent_mod.run_domain_agent(...)` |
| 95 | Append output to `state.agent_outputs[agent_name]` |
| 98 | Set `discussion_rounds_completed = round_number` |
| 100 | Stash outputs in `state._latest_round_outputs` for summarizer |

**Domain agents executed (in order):**  
`geo_strategy` → `economy_technology` → `domestic_ideology` → `security_taiwan` → `historical_analogy`  
(defined in `app/schemas.py:36–42`)

#### run_domain_agent() — `app/agents.py:191–245`

| Line | Logic |
|---|---|
| 201–202 | Validates agent name against `AGENT_SYSTEM_PROMPTS` (lines 70–102) |
| 204–209 | System prompt = agent role + safety + JSON instructions |
| 211–215 | Serialize previous discussion summary to JSON string (or empty in round 1) |
| 217–228 | User prompt = seed + mode + evidence blob + round number + prev summary + own prev position + schema hint |
| 229 | Truncate user prompt to `MAX_AGENT_INPUT_CHARS` |
| 232–244 | `llm.call_llm_json(...)` with per-agent `agent_name` and `round_number` |
| 245 | `_to_agent_output(...)` (lines 573–604) — coerce JSON dict → `AgentOutput` |

**Each domain agent makes 1 LLM call per round → up to 15 calls total (5 agents × 3 rounds).**

### 7b. Orchestrator summarize round (runs after each discussion round)

| | |
|---|---|
| **Core function** | `app/graph.py:122–137` → `_summarize_round(state, llm, round_number)` |
| **Wrappers** | `orchestrator_summarize_round_1/2/3` (lines 140–153) |

| Line | Logic |
|---|---|
| 123–125 | Read `_latest_round_outputs`; return if empty |
| 126–132 | Call `agent_mod.run_orchestrator_summary(...)` |
| 133 | Append result to `state.discussion_rounds` |
| 135–136 | If round ≥ 2 and `should_stop_early()` → set `_early_stopped = True` |

#### run_orchestrator_summary() — `app/agents.py:321–391`

| Line | Logic |
|---|---|
| 335–341 | Build compact text blob of each agent's main assessment + drivers |
| 343–363 | System + user prompts asking for compressed round summary JSON |
| 366 | Build heuristic fallback via `build_discussion_summary()` (`cost_control.py:57–102`) |
| 367–375 | `llm.call_llm_json(...)`, `agent_name="orchestrator_summary"` |
| 377–391 | Parse into `DiscussionSummary`; on failure return heuristic |

#### should_stop_early() — `app/cost_control.py:105–119`

Returns `True` when:
- `round_number >= 2`, AND
- `areas_of_disagreement` has ≤ 1 item, AND
- `emerging_timeline` has ≥ 3 items

When true, round 3 agents are skipped (`graph.py:70–71`, `115–116`).

**Up to 3 LLM calls for round summaries.**

---

## 8. Node ⑨ red_team_agent

| | |
|---|---|
| **Graph node** | `app/graph.py:156–168` → `red_team_node()` |
| **Agent function** | `app/agents.py:251–315` → `run_red_team_agent()` |

| Step | File:Lines | Logic |
|---|---|---|
| Get last discussion summary | `graph.py:157` | `state.discussion_rounds[-1]` |
| Compact evidence | `graph.py:158` | `compact_evidence_for_agents(...)` |
| Build prompts | `agents.py:259–282` | Red-team role: challenge assumptions, find gaps |
| LLM call | `agents.py:293–301` | `agent_name="red_team"`, `round_number=99` |
| Parse findings | `agents.py:303–314` | `_to_agent_output()` + loop `findings[]` → `RedTeamFinding` |
| Store | `graph.py:166–167` | Append to `agent_outputs["red_team"]`, set `red_team_findings` |

**1 LLM call.**

---

## 9. Node ⑩ orchestrator_synthesis

| | |
|---|---|
| **Graph node** | `app/graph.py:171–203` → `orchestrator_synthesis()` |

| Line | Logic |
|---|---|
| 172–177 | Collect each domain agent's **last** output from `agent_outputs` |
| 178–181 | Get red-team's last `AgentOutput` |
| 183–192 | `run_orchestrator_final_synthesis(...)` — **LLM call** for title, summary, disagreements, image prompt |
| 194–200 | Write synthesis fields into state; fallback image prompt via `build_image_prompt()` if empty |
| 202 | `build_final_timeline(last_per_agent)` — **no LLM**, deterministic merge |

#### run_orchestrator_final_synthesis() — `app/agents.py:394–482`

| Line | Logic |
|---|---|
| 411–417 | System prompt: synthesize one plausible scenario, not a prediction |
| 419–449 | User prompt: seed, evidence, final discussion summary, agent positions, red-team findings |
| 463–471 | `llm.call_llm_json(...)`, `agent_name="orchestrator_final"` |
| 473–482 | Return dict with title, summary, event_status, disagreements, image_prompt |

#### build_final_timeline() — `app/agents.py:509–553`

| Line | Logic |
|---|---|
| 513–519 | Map agent names → domain labels (strategy, economy, security, …) |
| 520–521 | Initialize buckets for years 2026–2031 (`SIMULATION_YEARS` in `utils.py:12`) |
| 523–543 | Loop all agents' `timeline_contributions`; dedupe by event text per year |
| 545–552 | Build `YearBlock` per year; headline = highest-probability event |

**1 LLM call + 1 deterministic timeline build.**

---

## 10. Node ⑪ orchestrator_image_generation

| | |
|---|---|
| **Graph node** | `app/graph.py:206–211` → `orchestrator_image_generation()` |
| **Image module** | `app/image_generation.py` |

| Step | File:Lines | Function | Logic |
|---|---|---|---|
| Ensure prompt exists | `graph.py:207–208` | inline | Calls `build_image_prompt()` if synthesis didn't produce one |
| Generate image | `graph.py:209` | `generate_image(run_id, prompt)` | `image_generation.py:38–100` |
| Store result | `graph.py:210` | inline | `state.image_result = result` |

#### generate_image() — `app/image_generation.py:38–100`

| Line | Logic |
|---|---|
| 49–51 | If `ENABLE_IMAGE_GENERATION=false` → return disabled `ImageResult` |
| 53–55 | Output path: `data/generated_images/<run_id>.png` |
| 57–66 | **Mock mode:** write 1×1 placeholder PNG |
| 68–100 | **Live mode:** OpenAI `images.generate()`; save base64 or URL; catch all errors |

**Never raises** — failures go into `image_result.error`.

---

## 11. Node ⑫ save_run

| | |
|---|---|
| **Graph node** | `app/graph.py:214–226` → `save_run_node()` |

| Line | Logic |
|---|---|
| 215 | `build_final_scenario(state)` — assemble public JSON shape |
| 217–223 | `db.save_scenario_run(...)` — first SQLite write |
| 224–225 | On failure, append to `state.errors` (run continues) |

#### build_final_scenario() — `app/graph.py:232–269`

| Line | Logic |
|---|---|
| 233–240 | Extract last assessment from each agent → `agent_summaries` dict |
| 242 | Red-team warnings from `red_team_findings[].issue` |
| 248–251 | `key_assumptions` from agents' `agreements` fields |
| 253–269 | Construct and return `FinalScenario` Pydantic model |

#### save_scenario_run() — `app/db.py:71–96`

| Line | Logic |
|---|---|
| 79 | Ensures DB exists |
| 81–95 | `INSERT OR REPLACE INTO scenario_runs` with full JSON blob |

---

## 12. Post-graph: metrics + second save + return

Back in `run_graph()` after all nodes finish:

| File:Lines | Logic |
|---|---|
| `graph.py:360` | `elapsed_seconds = now - start` |
| `graph.py:361–365` | Copy LLM metrics from `llm.metrics` into state |
| `graph.py:367` | Rebuild `FinalScenario` (now includes final metrics) |
| `graph.py:369–376` | **Second** `db.save_scenario_run()` — overwrites with metrics-filled JSON |
| `graph.py:380` | Return to `main.py:87` |
| `main.py:88` | `final.model_dump()` → HTTP JSON response |

---

## 13. Shared subsystem: LLMClient.call_llm_json

**Every agent** eventually calls this.  
**File:** `app/llm.py`  
**Function:** `call_llm_json()` — lines **90–133**

```
call_llm_json(system, user, agent_name, round_number, ...)
    │
    ├─ _cache_key()                    llm.py:137–152
    │     stable_hash(model, agent, round, prompts, context)   utils.py:23–29
    │
    ├─ db.cache_get(cache_key)         llm.py:104–108, db.py:136–150
    │     └── HIT → return cached JSON, increment cache_hits
    │
    ├─ CONFIG.mock_mode?               llm.py:110–112
    │     └── YES → _mock_json()       llm.py:226–379 (deterministic stubs)
    │
    └─ _call_openai_text()             llm.py:162–194
          ├── OpenAI(api_key)          llm.py:154–160
          ├── model = CONFIG.openai_model
          ├── response_format = json_object
          ├── retry up to 2 times with backoff
          └── extract_json(raw)        utils.py:43–73
                └── fallback dict if parse fails
    │
    └─ db.cache_set(...)               llm.py:131–132, db.py:153–175
    └─ metrics.record_call(...)        llm.py:125–129
```

### Mock responses by agent — `llm.py:226–379`

| Condition | Mock content |
|---|---|
| `agent_name == "evidence_rag"` | Evidence summary with hypothetical assumption note |
| `agent_name == "orchestrator_summary"` | Round discussion summary |
| `agent_name == "red_team"` | Critique + findings array |
| `agent_name == "orchestrator_final"` | Title, summary, image prompt |
| Any domain agent | Assessment + timeline contributions for 2026/2028/2030 |

---

## 14. Shared subsystem: RAG retrieval

### Offline ingestion (before runtime)

| File | Lines | Function | Logic |
|---|---|---|---|
| `scripts/ingest_docs.py` | 28–29 | `main()` | CLI entry |
| `scripts/ingest_docs.py` | 28 | `ingest_knowledge_base()` | Delegates to rag module |
| `app/rag.py` | 110–151 | `ingest_knowledge_base()` | Walk `knowledge_base/`, chunk, tag metadata, write JSON |

### Runtime retrieval (during Evidence agent)

| Step | File:Lines | Function | Logic |
|---|---|---|---|
| Cache check | `rag.py:248–252` | `retrieve_with_cache()` | In-memory dict keyed by `hash(seed + mode)` |
| Load chunks | `rag.py:154–165` | `_load_chunks()` | Read `data/rag_chunks.json` (empty list if missing) |
| Score documents | `rag.py:214–218` | `_tfidf_scores()` or `_keyword_score()` | TF-IDF preferred; keyword fallback |
| Rank + truncate | `rag.py:220–236` | `retrieve()` | Top-K chunks → `EvidenceChunk` objects |
| Return | `rag.py:253–255` | `retrieve_with_cache()` | `(chunks, cache_hit_bool)` |

**Empty KB is safe:** `retrieve()` returns `[]` at line 211–212; Evidence agent still runs.

---

## 15. Shared subsystem: cost control helpers

| Function | File:Lines | Called from | Purpose |
|---|---|---|---|
| `compact_evidence_for_agents()` | `cost_control.py:21–44` | `graph.py:73`, `graph.py:158` | One paragraph for all agents; max `MAX_EVIDENCE_CHARS` |
| `compact_agent_position()` | `cost_control.py:47–54` | `graph.py:83` | One line of agent's prior position for next round |
| `build_discussion_summary()` | `cost_control.py:57–102` | `agents.py:366` | Heuristic fallback for round summary |
| `should_stop_early()` | `cost_control.py:105–119` | `graph.py:135` | Skip round 3 when consensus is strong |
| `truncate()` | `utils.py:32–37` | Many files | Hard cap on string length |

---

## 16. Frontend: render result

After JSON returns to browser:

| Step | File:Lines | Function | Logic |
|---|---|---|---|
| Show result panel | `app.js:84–85` | `renderResult()` | Hide empty state, show `#result` |
| Title + badge | `app.js:87–92` | `renderResult()` | Scenario title + event_status badge |
| Timeline cards | `app.js:94`, `104–133` | `renderTimeline()` | One card per year 2026–2031 |
| Discussion rounds | `app.js:95`, `135–155` | `renderDiscussion()` | Agree / disagree / uncertain per round |
| Agent summaries | `app.js:96`, `157–176` | `renderAgentSummaries()` | Last assessment per agent |
| Disagreements | `app.js:97`, `178–186` | `renderList()` | Bullet list |
| Red-team warnings | `app.js:98` | `renderList()` | Bullet list |
| Image | `app.js:99`, `192–218` | `renderImage()` | `<img src="/generated_images/<run_id>.png">` |
| Metrics | `app.js:100`, `220–239` | `renderMetrics()` | LLM calls, cache hits, elapsed, etc. |

---

## 17. Quick reference: all LLM calls per run

In **live mode**, a full run (3 discussion rounds, no early stop, no cache hits) makes approximately:

| # | agent_name | When | File:Line |
|---|---|---|---|
| 1 | `evidence_rag` | Node ② | `agents.py:154` |
| 2–6 | `geo_strategy` … `historical_analogy` | Round 1 | `agents.py:232` (×5) |
| 7 | `orchestrator_summary` | After round 1 | `agents.py:367` |
| 8–12 | domain agents | Round 2 | `agents.py:232` (×5) |
| 13 | `orchestrator_summary` | After round 2 | `agents.py:367` |
| 14–18 | domain agents | Round 3 (if not early-stopped) | `agents.py:232` (×5) |
| 19 | `orchestrator_summary` | After round 3 | `agents.py:367` |
| 20 | `red_team` | Node ⑨ | `agents.py:293` |
| 21 | `orchestrator_final` | Node ⑩ | `agents.py:463` |

**Typical total: ~15–21 text LLM calls** (15 if early stop after round 2 skips round 3 agents + summary).

Plus **1 image API call** if `ENABLE_IMAGE_GENERATION=true` and API key present (`image_generation.py:72`).

With **cache hits**, repeated seeds skip most of these (`llm.py:104–108`).

---

## State object field lifecycle

Track where each important field gets written:

| Field | First set | Final set |
|---|---|---|
| `run_id` | `graph.py:342` (`new_run_id`) | unchanged |
| `event_status` | `graph.py:47` (`classify_event_status`) | `graph.py:196` (synthesis may override) |
| `evidence_summary` | `graph.py:60` | unchanged |
| `agent_outputs` | `graph.py:95` (each round appends) | `graph.py:166` (red_team) |
| `discussion_rounds` | `graph.py:133` (each summarize) | unchanged |
| `_early_stopped` | `graph.py:136` | unchanged |
| `red_team_findings` | `graph.py:167` | unchanged |
| `scenario_title` | `graph.py:49` (placeholder) | `graph.py:194` (synthesis) |
| `scenario_summary` | — | `graph.py:195` |
| `final_timeline` | — | `graph.py:202` |
| `image_prompt` | — | `graph.py:198` |
| `image_result` | — | `graph.py:210` |
| `run_metrics` | `graph.py:343` | `graph.py:360–365` |

---

## Related files

| File | Role |
|---|---|
| `app/schemas.py` | All Pydantic types (`ScenarioState`, `AgentOutput`, `FinalScenario`, …) |
| `app/utils.py` | IDs, hashing, JSON extraction, truncation |
| `app/config.py` | Environment configuration |
| `frontend/index.html` | Dashboard layout |
| `frontend/style.css` | Dark theme styles |
| `tests/conftest.py` | Forces mock mode + temp DB for all tests |

---

## How to debug a single step

1. Add a `print()` or breakpoint inside the graph node in `app/graph.py`.
2. Or add one inside the agent function in `app/agents.py`.
3. Run one simulation: `POST /api/run-scenario` from the UI or `tests/test_api.py`.
4. Check `state.errors` in the response for non-fatal node failures.
5. Inspect SQLite: `data/scenarios.sqlite` → table `scenario_runs` → column `full_json`.

For mock-mode testing without OpenAI costs, unset `OPENAI_API_KEY` in `.env` — all LLM calls route to `_mock_json()` at `llm.py:226`.
