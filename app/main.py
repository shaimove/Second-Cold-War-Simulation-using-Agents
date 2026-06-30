"""FastAPI entrypoint.

Serves the static dashboard from `frontend/` and exposes the
scenario API. Saved runs and generated images are stored under `data/`.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config as _config_mod
from . import db
from .graph import run_graph, resume_graph
from .checkpoints import get_checkpoint_status
from .monitor.service import finalize_run_payload, judge_saved_run
from .rag import ingest_knowledge_base
from .schemas import ScenarioRequest, SavedRunSummary


FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
GENERATED_IMAGES_DIR = os.path.abspath(_config_mod.CONFIG.generated_images_dir)
os.makedirs(GENERATED_IMAGES_DIR, exist_ok=True)


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    db.init_db()
    yield


app = FastAPI(
    title="Cold War Scenario Simulator", version="1.0.0", lifespan=_lifespan
)

app.mount(
    "/generated_images",
    StaticFiles(directory=GENERATED_IMAGES_DIR),
    name="generated_images",
)


# --- Static frontend -------------------------------------------------------


@app.get("/")
def serve_index() -> FileResponse:
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/style.css")
def serve_css() -> FileResponse:
    return FileResponse(os.path.join(FRONTEND_DIR, "style.css"), media_type="text/css")


@app.get("/app.js")
def serve_js() -> FileResponse:
    return FileResponse(
        os.path.join(FRONTEND_DIR, "app.js"), media_type="application/javascript"
    )


# --- API -------------------------------------------------------------------


@app.get("/api/config")
def get_config() -> Dict[str, Any]:
    """Lightweight, safe subset of config (no API key) for the frontend."""
    cfg = _config_mod.CONFIG
    return {
        "model": cfg.openai_model,
        "orchestrator_model": cfg.openai_orchestrator_model,
        "image_model": cfg.openai_image_model,
        "mock_mode": cfg.mock_mode,
        "use_rag": cfg.use_rag,
        "use_llm_cache": cfg.use_llm_cache,
        "enable_image_generation": cfg.enable_image_generation,
        "max_agent_discussion_rounds": cfg.max_agent_discussion_rounds,
        "judge_model": cfg.openai_judge_model,
        "enable_run_judge": cfg.enable_run_judge,
        "enable_timeline_judge": cfg.enable_timeline_judge,
        "enable_graph_checkpoints": cfg.enable_graph_checkpoints,
    }


@app.post("/api/run-scenario")
def run_scenario(req: ScenarioRequest) -> Dict[str, Any]:
    if not req.seed or not req.seed.strip():
        raise HTTPException(status_code=400, detail="seed must not be empty")
    final = run_graph(seed=req.seed.strip(), scenario_mode=req.scenario_mode)
    return finalize_run_payload(final.model_dump())


@app.get("/api/runs")
def list_runs() -> List[Dict[str, Any]]:
    rows = db.list_scenario_runs(limit=100)
    return [SavedRunSummary(**r).model_dump() for r in rows]


@app.get("/api/runs/{run_id}/checkpoint")
def get_run_checkpoint(run_id: str) -> Dict[str, Any]:
    status = get_checkpoint_status(run_id)
    if status is None:
        raise HTTPException(status_code=404, detail="checkpoint not found")
    return status


@app.post("/api/runs/{run_id}/resume")
def resume_run(run_id: str) -> Dict[str, Any]:
    try:
        final = resume_graph(run_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return finalize_run_payload(final.model_dump())


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> JSONResponse:
    payload = db.load_scenario_run(run_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="run not found")
    return JSONResponse(payload)


@app.post("/api/runs/{run_id}/judge")
def judge_run(run_id: str) -> Dict[str, Any]:
    """Run hard gates + LLM quality judge on a saved run."""
    updated = judge_saved_run(run_id)
    if updated is None:
        raise HTTPException(status_code=404, detail="run not found")
    return updated


@app.post("/api/ingest")
def ingest() -> Dict[str, Any]:
    result = ingest_knowledge_base()
    return {
        "chunk_count": result.chunk_count,
        "files_processed": result.files_processed,
        "pdf_files": result.pdf_files,
        "text_files": result.text_files,
        "skipped_files": result.skipped_files,
        "output_path": result.output_path,
        "vector_index_path": result.vector_index_path,
    }
