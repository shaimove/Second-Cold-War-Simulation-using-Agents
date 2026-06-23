"""LangGraph checkpoint persistence tests."""
from app.checkpoints import get_checkpoint_status, get_checkpointer, graph_run_config, reset_checkpointer_cache, set_active_llm, reset_active_llm
from app.graph import NODES, build_graph, resume_graph, run_graph
from app.llm import LLMClient
from app.schemas import RunMetrics, ScenarioState
from app.utils import new_run_id


def test_run_graph_creates_checkpoint_file(tmp_path, monkeypatch):
    cp_path = tmp_path / "checkpoints.sqlite"
    monkeypatch.setenv("CHECKPOINT_SQLITE_PATH", str(cp_path))
    monkeypatch.setenv("ENABLE_GRAPH_CHECKPOINTS", "true")
    from app import config as cfg_mod

    cfg_mod.CONFIG = cfg_mod.load_config()
    reset_checkpointer_cache()

    final = run_graph(seed="checkpoint smoke test", scenario_mode="base_case")
    assert final.run_id
    assert cp_path.exists()

    status = get_checkpoint_status(final.run_id)
    assert status is not None
    assert status["can_resume"] is False


def test_checkpoint_resume_after_interrupt(tmp_path, monkeypatch):
    cp_path = tmp_path / "checkpoints.sqlite"
    monkeypatch.setenv("CHECKPOINT_SQLITE_PATH", str(cp_path))
    monkeypatch.setenv("ENABLE_GRAPH_CHECKPOINTS", "true")
    from app import config as cfg_mod

    cfg_mod.CONFIG = cfg_mod.load_config()
    reset_checkpointer_cache()

    checkpointer = get_checkpointer()
    assert checkpointer is not None

    stop_after = "evidence_rag_agent"
    compiled = build_graph(
        checkpointer=checkpointer,
        interrupt_after=[stop_after],
    )
    assert compiled is not None

    run_id = new_run_id()
    llm = LLMClient()
    state = ScenarioState(
        run_id=run_id,
        seed="resume after evidence step",
        scenario_mode="base_case",
    )
    state.run_metrics = RunMetrics()
    config = graph_run_config(run_id)
    token = set_active_llm(llm)
    try:
        from app.graph import _state_to_dict

        compiled.invoke(_state_to_dict(state), config=config)
    finally:
        reset_active_llm(token)

    status = get_checkpoint_status(run_id)
    assert status is not None
    assert status["can_resume"] is True
    assert status["next_nodes"]
    assert status["next_nodes"][0] != stop_after

    final = resume_graph(run_id, llm=llm)
    assert final.run_id == run_id
    assert final.scenario_summary
    assert final.run_metrics.discussion_rounds_completed >= 1

    status_after = get_checkpoint_status(run_id)
    assert status_after is not None
    assert status_after["can_resume"] is False


def test_resume_unknown_run_raises():
    try:
        resume_graph("run_does_not_exist")
    except ValueError as e:
        assert "No checkpoint found" in str(e)
    else:
        raise AssertionError("expected ValueError")
