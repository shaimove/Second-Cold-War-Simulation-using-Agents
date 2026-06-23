import os

from app import config as _config_mod
from app.image_generation import build_image_prompt, generate_image

CONFIG = _config_mod.CONFIG


def test_build_image_prompt_includes_safety_tail():
    prompt = build_image_prompt("Title", "Summary")
    assert "Non-graphic" in prompt
    assert "no operational" in prompt.lower()


def test_mock_image_generation_writes_placeholder():
    result = generate_image("test_run", "some prompt")
    assert result.enabled is True
    assert result.generated is True
    assert result.mock is True
    assert result.path is not None
    assert os.path.exists(result.path)


def _make_config(**overrides):
    base = _config_mod.CONFIG
    fields = dict(
        openai_api_key=base.openai_api_key,
        openai_model=base.openai_model,
        openai_orchestrator_model=base.openai_orchestrator_model,
        openai_judge_model=base.openai_judge_model,
        enable_run_judge=base.enable_run_judge,
        openai_image_model=base.openai_image_model,
        use_rag=base.use_rag,
        use_llm_cache=False,
        enable_image_generation=base.enable_image_generation,
        parallel_domain_agents=base.parallel_domain_agents,
        max_agent_discussion_rounds=base.max_agent_discussion_rounds,
        max_retrieved_docs=base.max_retrieved_docs,
        max_agent_input_chars=base.max_agent_input_chars,
        max_evidence_chars=base.max_evidence_chars,
        self_position_max_chars=base.self_position_max_chars,
        enable_graph_checkpoints=base.enable_graph_checkpoints,
        sqlite_path=base.sqlite_path,
        checkpoint_sqlite_path=base.checkpoint_sqlite_path,
        rag_chunks_path=base.rag_chunks_path,
        generated_images_dir=base.generated_images_dir,
    )
    fields.update(overrides)
    return base.__class__(**fields)


def test_image_generation_failure_does_not_raise(monkeypatch):
    import sys
    from types import ModuleType

    from app import image_generation as ig

    class _FakeImages:
        def generate(self, **_kwargs):
            raise RuntimeError("forced failure")

    class _FakeClient:
        def __init__(self, **_kwargs):
            self.images = _FakeImages()

    fake_cfg = _make_config(openai_api_key="sk-test", enable_image_generation=True)
    fake_openai = ModuleType("openai")
    fake_openai.OpenAI = _FakeClient
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    result = ig.generate_image("crash_run", "prompt", config=fake_cfg)
    assert result.error is not None
    assert result.generated is False


def test_image_disabled_returns_disabled_result():
    from app import image_generation as ig

    fake_cfg = _make_config(openai_api_key=None, enable_image_generation=False)
    result = ig.generate_image("r", "p", config=fake_cfg)
    assert result.enabled is False
    assert result.generated is False
