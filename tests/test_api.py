from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_index_serves_html():
    r = client.get("/")
    assert r.status_code == 200
    assert "<html" in r.text.lower()


def test_static_assets():
    r1 = client.get("/style.css")
    r2 = client.get("/app.js")
    assert r1.status_code == 200
    assert r2.status_code == 200


def test_config_endpoint():
    r = client.get("/api/config")
    assert r.status_code == 200
    body = r.json()
    assert "model" in body
    assert "mock_mode" in body


def test_run_scenario_returns_valid_json_in_mock_mode():
    r = client.post(
        "/api/run-scenario",
        json={"seed": "Taiwan tension rises after election.", "scenario_mode": "base_case"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    for key in (
        "run_id",
        "scenario_title",
        "timeline",
        "discussion_summary",
        "run_metrics",
        "image",
        "monitor",
    ):
        assert key in body
    assert len(body["timeline"]) == 6
    monitor = body["monitor"]
    assert monitor["gates"]["passed"] is True
    assert len(monitor["gates"]["checks"]) == 6


def test_run_scenario_rejects_empty_seed():
    r = client.post(
        "/api/run-scenario", json={"seed": "  ", "scenario_mode": "base_case"}
    )
    assert r.status_code == 400


def test_runs_list_and_get():
    # Create a run first.
    r = client.post(
        "/api/run-scenario", json={"seed": "x", "scenario_mode": "base_case"}
    )
    assert r.status_code == 200
    run_id = r.json()["run_id"]

    lst = client.get("/api/runs").json()
    assert any(item["run_id"] == run_id for item in lst)

    full = client.get("/api/runs/" + run_id).json()
    assert full["run_id"] == run_id

    missing = client.get("/api/runs/does_not_exist")
    assert missing.status_code == 404


def test_ingest_endpoint_runs():
    r = client.post("/api/ingest")
    assert r.status_code == 200
    body = r.json()
    assert "chunk_count" in body
