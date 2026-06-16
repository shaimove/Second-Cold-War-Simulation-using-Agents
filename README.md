# Cold War Scenario Simulator using Agentic System

A local multi-agent AI application that generates a five-year (2026–2031)
plausible USA–China rivalry scenario from a single seed sentence. Built as a
portfolio project to demonstrate **LangGraph orchestration, multi-agent
discussion, RAG, structured outputs, image generation, SQLite caching, and
unit-tested production-style Python**.

> Disclaimer: this app produces *one plausible scenario*, not a prediction.
> It stays at the strategic / scenario-planning level and explicitly avoids
> operational military detail, targeting, or instructions for real-world
> harm. See the safety constraint inside `app/agents.py`.


**See a real example run** (rendered exactly like the live dashboard, no setup required):
[examples/example_scenario.html](https://htmlpreview.github.io/?https://github.com/shaimove/Second-Cold-War-Simulation-using-Agents/blob/main/examples/example_scenario.html)

**Understand the code pipeline:** [docs/PIPELINE.md](docs/PIPELINE.md)  
**Deep dive (file + line + function per step):** [docs/PIPELINE_CODE_WALKTHROUGH.md](docs/PIPELINE_CODE_WALKTHROUGH.md)  
**Interview algo spec (design, prompts, tuning):** [docs/ALGO_SPEC.md](docs/ALGO_SPEC.md)  
**Quality monitor MVP (gates + LLM judge):** [docs/MONITOR.md](docs/MONITOR.md)  
**Full evaluation reference:** [docs/EVALUATION.md](docs/EVALUATION.md)  
**Interview story (architecture, agents, cost, fallbacks):** [docs/INTERVIEW.md](docs/INTERVIEW.md)

---

## Why this is an "agentic" AI system

The app is not a single prompt. It is a graph of specialized agents that
collaborate, disagree, revise their positions, and get critiqued by a
red-team before a final orchestrator synthesizes them:

- **Orchestrator** initializes state, summarizes each round, decides when
  to stop, and produces the final 2026–2031 timeline + image prompt.
- **Evidence / RAG agent** pulls relevant context from local
  markdown/text knowledge base, separating *observed facts* from
  *historical analogies*, *strategy frameworks*, and *hypothetical
  assumptions* extracted from the seed.
- **Five domain agents** (Geo-Strategy, Economy & Technology, Domestic
  Politics & Ideology, Security/Taiwan, Historical Analogy) each give an
  independent assessment, then revise across up to **three discussion
  rounds**.
- **Red-Team agent** challenges the consensus.
- **Image generation** produces a non-graphic editorial illustration.

Every run is deterministic-by-cache and works **with or without** an
OpenAI API key (mock mode is the default when no key is set).

---

## Architecture

```mermaid
flowchart TD
    A[User Seed + Mode] --> B[Orchestrator Init]
    B --> C[Evidence / RAG Agent]
    C --> D1[Discussion Round 1<br/>5 domain agents]
    D1 --> S1[Orchestrator Summary 1]
    S1 --> D2[Discussion Round 2]
    D2 --> S2[Orchestrator Summary 2]
    S2 --> D3[Discussion Round 3]
    D3 --> S3[Orchestrator Summary 3]
    S3 --> R[Red-Team Agent]
    R --> F[Orchestrator Final Synthesis]
    F --> I[Image Generation]
    I --> SV[Save Run to SQLite]
    SV --> Z[Final JSON to Frontend]
```

### LangGraph flow

The graph is built in `app/graph.py` using `langgraph.StateGraph`. If
LangGraph isn't installed or its API drifts, the same nodes execute via
a deterministic sequential fallback (see `_run_sequential`). The tests
exercise both paths.

```
START
 -> orchestrator_initialize
 -> evidence_rag_agent
 -> discussion_round_1 -> orchestrator_summarize_round_1
 -> discussion_round_2 -> orchestrator_summarize_round_2
 -> discussion_round_3 -> orchestrator_summarize_round_3
 -> red_team_agent
 -> orchestrator_synthesis
 -> orchestrator_image_generation
 -> save_run
END
```

### Multi-round discussion design

In each round every domain agent receives only:

- the seed and scenario mode
- a **compact evidence summary** (never the raw documents)
- a **compact summary of the previous round** (not all prior outputs)
- its own previous position
- a strict JSON output schema

This keeps token usage flat as rounds grow. After round 2 the Orchestrator
may short-circuit to round 3 if disagreements collapse (see
`cost_control.should_stop_early`). The default still supports up to 3
rounds.

---

## RAG

The knowledge base is a folder of `.md`, `.txt`, and `.pdf` files in
`knowledge_base/`. `scripts/ingest_docs.py` extracts PDF text (cached
under `data/preprocessed/`), chunks all documents, infers a
`source_type` (`current_context | historical_analogy | strategy_framework
| unknown`) from the path, and writes them to `data/rag_chunks.json`.

Retrieval uses **TF-IDF cosine similarity** (scikit-learn) when available
and a **keyword-overlap fallback** when it isn't. The whole pipeline is
empty-safe: if you haven't added any books yet, the Evidence agent
simply reports that future events are hypothetical and uses general
model reasoning.

To add books later:

```
knowledge_base/
  history/cold_war_overview.md
  strategy/containment_doctrine.md
  current_context/2025_chip_controls.txt
python scripts/ingest_docs.py
```

or hit `POST /api/ingest` while the app is running.

### Tier 2 RAG upgrade

Agents do **not** read the entire knowledge base. One shared index is
searched with **metadata filters** (domain, source type, period inferred
from folder and filename). Each specialist receives a **targeted evidence
packet**:

- **Baseline retrieval** at the Evidence step (broad seed query).
- **Round 1:** per-agent retrieval (3–4 chunks) plus role-specific evidence
  lanes (observed, historical, economy, security, etc.).
- **Round 2:** shared **disagreement retrieval** from the Orchestrator’s
  round-1 summary (no second per-agent search).
- **Round 3:** reuses round-2 disagreement evidence by default (no new
  retrieval unless `RAG_ENABLE_ROUND3_NEW_RETRIEVAL=true`).
- **Red-Team:** critique-focused retrieval before the challenge pass.
- **Final Orchestrator:** compact `FinalEvidencePacket` (cited chunks,
  top lanes, dispute notes)—not raw full-corpus dumps.

Agents may cite only `chunk_id`s provided in their prompt; invalid citations
are stripped and recorded in `run_metrics.citation_warnings`. Tune behavior
via `app/rag_config.py` or env vars such as `RAG_AGENT_ROUND1_FINAL_K`,
`RAG_ENABLE_AGENT_RAG`, `RAG_MAX_FINAL_EVIDENCE_ITEMS`.

After changing PDFs or metadata rules, **re-run** `python scripts/ingest_docs.py`
so chunk `domain` tags are refreshed (load-time inference also applies).

### Parallel domain agents

Within each discussion round, the five domain agents run **concurrently**
via LangChain `RunnableParallel` (`app/parallel_agents.py`). Set
`PARALLEL_DOMAIN_AGENTS=false` in `.env` to fall back to sequential calls.

---

## Image generation

After final synthesis the Orchestrator produces an editorial-illustration
prompt and calls `OPENAI_IMAGE_MODEL`. Generated PNGs are saved under
`data/generated_images/<run_id>.png` and served at
`/generated_images/<run_id>.png`.

- Image failures **never crash a run** - the error is stored in
  `image.error` and shown in the UI.
- In mock mode (no API key) a tiny placeholder PNG is written so the
  frontend has something to render.
- The prompt enforces a **non-graphic, non-tactical** editorial style.

---

## Cost-control techniques

| Technique | Implementation |
|---|---|
| Evidence lanes + targeted packets | `rag.build_evidence_lanes`, `rag_citations.build_agent_evidence_packet` |
| Per-agent round-1 retrieval | `rag.retrieve_for_agent` |
| Disagreement retrieval (round 2) | `rag.retrieve_for_disagreement` |
| Round compaction | `cost_control.build_discussion_summary` + LLM compaction |
| Strict structured outputs | `response_format=json_object` + Pydantic coercion |
| Early stopping | `cost_control.should_stop_early` |
| LLM cache | SQLite `llm_cache` table keyed by hash(model+agent+context) |
| Retrieval cache | In-process dict keyed by hash(seed+mode+agent+round+query+filters) |
| RAG metrics | `rag_calls`, `unique_chunks_used`, `most_cited_chunk_ids`, etc. |
| Token budget | `MAX_AGENT_INPUT_CHARS`, `MAX_EVIDENCE_CHARS` |
| Mock mode | Deterministic stub responses when no API key |
| Run metrics | LLM calls, cache hits, retrieved docs, elapsed, est. tokens |

---

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# (optional) add OPENAI_API_KEY=... to .env to enable live mode
python scripts/ingest_docs.py
uvicorn app.main:app --reload
```

Then open <http://localhost:8000>.

Convenience wrapper: `bash run_local.sh`.

### Configuration (`.env`)

```
OPENAI_API_KEY=                 # empty = mock mode
OPENAI_MODEL=gpt-5.4-mini             # domain agents, evidence, red-team
OPENAI_ORCHESTRATOR_MODEL=gpt-5.4   # round summaries + final synthesis + JSON repair
OPENAI_IMAGE_MODEL=gpt-image-2      # image model

USE_RAG=true
USE_LLM_CACHE=true
ENABLE_IMAGE_GENERATION=true

MAX_AGENT_DISCUSSION_ROUNDS=3
MAX_RETRIEVED_DOCS=5
MAX_AGENT_INPUT_CHARS=6000
MAX_EVIDENCE_CHARS=2500
```

The model is **never hard-coded**; everything flows through `app/config.py`.

---

## Tests

```bash
pytest
```

All tests run **without any OpenAI calls**:

- API key is stripped via `tests/conftest.py`
- Mock LLM responses are schema-faithful
- Each test uses a fresh temp SQLite + temp RAG path
- The image generator writes a placeholder PNG in mock mode

Test files:

| File | Covers |
|---|---|
| `test_config.py` | env loading, bool/int parsing, defaults |
| `test_schemas.py` | Pydantic validation + invalid-mode rejection |
| `test_rag.py` | empty KB, ingest, retrieve, retrieval cache |
| `test_llm.py` | mock mode, cache key stability, JSON fallback |
| `test_agents.py` | all agents return required fields, safety prompt |
| `test_graph.py` | end-to-end run, all 3 rounds, 2026–2031 coverage |
| `test_db.py` | SQLite init, save/load run, cache set/get |
| `test_api.py` | static serving, run-scenario, runs list/get, ingest |
| `test_image_generation.py` | mock placeholder, failure-tolerant, disabled |

---

## Example seeds

- "China enters a major financial crisis after a real-estate banking shock."
- "The U.S. announces a new wave of AI chip export controls."
- "Taiwan elects a more independence-leaning government."
- "A major cyberattack disrupts global semiconductor supply chains."
- "China and the U.S. unexpectedly restart high-level trade negotiations."

---

## API surface

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Dashboard |
| `GET` | `/style.css`, `/app.js` | Static assets |
| `GET` | `/generated_images/{file}` | Image file |
| `GET` | `/api/config` | Public config (no secrets) |
| `POST` | `/api/run-scenario` | Run multi-agent simulation |
| `GET` | `/api/runs` | List saved runs |
| `GET` | `/api/runs/{run_id}` | Load a saved run |
| `POST` | `/api/ingest` | Re-ingest knowledge base |

---

## Limitations

- No live news API: the "current context" comes only from your local
  knowledge base.
- The model is asked to phrase outputs as *one plausible scenario* - it
  is not predicting the future.
- Retrieval is intentionally simple (TF-IDF / keyword overlap). For
  production you'd swap in pgvector / Chroma / Qdrant.
- No auth, no rate limits - local-only by design.

## Future improvements

- Real vector store (pgvector / Chroma) behind the same `retrieve` API.
- Async agent execution inside each round.
- Streaming progress via Server-Sent Events.
- React frontend with a saved-run diff view.
- Live news ingestion job.
- Per-agent token budgets with adaptive truncation.
