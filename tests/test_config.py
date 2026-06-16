from app import config as cfg_mod


def test_load_config_defaults(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_MODEL", "foo-1")
    c = cfg_mod.load_config()
    assert c.openai_model == "foo-1"
    assert c.mock_mode is True
    assert c.max_agent_discussion_rounds >= 1
    assert c.self_position_max_chars == 1000


def test_load_config_reads_bools_and_ints(monkeypatch):
    monkeypatch.setenv("USE_RAG", "false")
    monkeypatch.setenv("USE_LLM_CACHE", "0")
    monkeypatch.setenv("MAX_AGENT_DISCUSSION_ROUNDS", "2")
    monkeypatch.setenv("SELF_POSITION_MAX_CHARS", "1200")
    c = cfg_mod.load_config()
    assert c.use_rag is False
    assert c.use_llm_cache is False
    assert c.max_agent_discussion_rounds == 2
    assert c.self_position_max_chars == 1200


def test_invalid_int_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("MAX_AGENT_DISCUSSION_ROUNDS", "not-a-number")
    c = cfg_mod.load_config()
    assert c.max_agent_discussion_rounds == 3


def test_mock_mode_when_no_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    c = cfg_mod.load_config()
    assert c.mock_mode is True
