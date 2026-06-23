"""API tests for checkpoint resume endpoints."""
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_checkpoint_status_404_for_unknown_run():
    r = client.get("/api/runs/run_missing/checkpoint")
    assert r.status_code == 404


def test_resume_unknown_run_returns_404():
    r = client.post("/api/runs/run_missing/resume")
    assert r.status_code == 404
