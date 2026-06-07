import os

from app import rag


def test_retrieve_with_empty_kb_returns_empty(tmp_path):
    out = tmp_path / "rag.json"
    rag.ingest_knowledge_base(str(tmp_path / "kb_does_not_exist"), str(out))
    assert os.path.exists(out)
    chunks = rag.retrieve("China chip controls", chunks_path=str(out))
    assert chunks == []


def test_ingest_and_retrieve_basic(tmp_path):
    kb = tmp_path / "kb"
    (kb / "strategy").mkdir(parents=True)
    (kb / "strategy" / "containment.md").write_text(
        "Containment was a U.S. strategy during the Cold War with the USSR. "
        "It shaped alliances and deterrence frameworks in the Indo-Pacific.",
        encoding="utf-8",
    )
    (kb / "history").mkdir(parents=True)
    (kb / "history" / "ussr_rivalry.txt").write_text(
        "The USSR rivalry produced arms races, sanctions, and proxy wars.",
        encoding="utf-8",
    )

    out = tmp_path / "chunks.json"
    res = rag.ingest_knowledge_base(str(kb), str(out))
    assert res.files_processed == 2
    assert res.chunk_count >= 2

    loaded = rag.load_chunks(str(out))
    assert loaded[0]["chunk_id"].startswith("kb_")

    chunks = rag.retrieve("Cold War alliances", chunks_path=str(out))
    assert len(chunks) > 0
    paths = [c.source_path for c in chunks]
    assert any("containment.md" in p for p in paths)
    for c in chunks:
        assert c.chunk_id
        assert c.source_type in (
            "book",
            "report",
            "framework",
            "current_context",
            "historical_analogy",
            "strategy_framework",
            "unknown",
        )
        assert c.domain in rag.VALID_DOMAINS


def test_retrieval_cache_hit(tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "doc.md").write_text(
        "Semiconductor supply chains and export controls.", encoding="utf-8"
    )
    out = tmp_path / "chunks.json"
    rag.ingest_knowledge_base(str(kb), str(out))

    rag.clear_retrieval_cache()
    cache = {}
    chunks1, hit1 = rag.retrieve_with_cache("chip controls", "base_case", cache=cache)
    chunks2, hit2 = rag.retrieve_with_cache("chip controls", "base_case", cache=cache)
    assert hit1 is False
    assert hit2 is True
    assert len(chunks1) == len(chunks2)


def test_ingest_pdf_uses_extractor(monkeypatch, tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    pdf_path = kb / "china_cold_war.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 placeholder")

    monkeypatch.setattr(
        rag,
        "_extract_pdf_text",
        lambda _p: (
            "Are the United States and China in a new Cold War? "
            "Containment and strategic rivalry echo historical patterns."
        ),
    )

    pre_dir = tmp_path / "preprocessed"
    out = tmp_path / "chunks.json"
    res = rag.ingest_knowledge_base(str(kb), str(out), preprocessed_dir=str(pre_dir))
    assert res.files_processed == 1
    assert res.pdf_files == 1
    assert res.chunk_count >= 1
    assert (pre_dir / "china_cold_war.txt").exists()

    chunks = rag.retrieve("USA China Cold War containment", chunks_path=str(out))
    assert len(chunks) > 0
    assert chunks[0].source_type in (
        "book",
        "historical_analogy",
        "current_context",
        "strategy_framework",
        "unknown",
    )
    assert "china_cold_war.pdf" in chunks[0].source_path
