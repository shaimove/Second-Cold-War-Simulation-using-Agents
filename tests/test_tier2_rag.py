"""Tier 2 RAG: metadata, lanes, per-agent retrieval, citations."""
import os

import pytest

from app import rag
from app.rag import (
    RagMetricsRecorder,
    RetrievalFilters,
    build_evidence_lanes,
    build_final_evidence_packet,
    clear_retrieval_cache,
    retrieve_baseline,
    retrieve_candidates,
    retrieve_for_agent,
    retrieve_for_disagreement,
    retrieve_for_red_team,
    rerank_candidates,
)
from app.rag_citations import (
    build_agent_evidence_packet,
    lanes_text_for_agent,
    validate_and_apply_citations,
)
from app.rag_config import ACTIVE_RAG_CONFIG
from app.schemas import (
    AgentOutput,
    DiscussionSummary,
    EvidenceChunk,
    EvidenceLanes,
    EvidenceSummary,
    RunMetrics,
)
from app.graph import run_graph, NODES


@pytest.fixture
def kb_with_samples(tmp_path):
    kb = tmp_path / "kb"
    (kb / "historical").mkdir(parents=True)
    (kb / "security").mkdir(parents=True)
    (kb / "economy").mkdir(parents=True)
    (kb / "historical" / "cold_war_ussr.txt").write_text(
        "The US USSR Cold War produced containment, arms races, and crisis management lessons.",
        encoding="utf-8",
    )
    (kb / "security" / "taiwan_deterrence.txt").write_text(
        "Taiwan deterrence and gray-zone escalation require crisis stability frameworks.",
        encoding="utf-8",
    )
    (kb / "economy" / "chip_trade.txt").write_text(
        "Semiconductor export controls and sanctions reshape supply chains and trade.",
        encoding="utf-8",
    )
    out = tmp_path / "chunks.json"
    rag.ingest_knowledge_base(str(kb), str(out))
    return str(out)


def test_ingest_chunk_metadata(kb_with_samples):
    chunks = rag.load_chunks(kb_with_samples)
    assert len(chunks) >= 3
    c0 = chunks[0]
    assert c0["chunk_id"].startswith("kb_")
    assert c0["source_name"]
    assert c0["char_count"] > 0
    assert c0["domain"] in rag.VALID_DOMAINS


def test_infer_historical_domain_from_path(tmp_path):
    path = str(tmp_path / "books" / "cold_war_ussr_analogy.pdf")
    dom = rag._infer_domain(path, "US USSR rivalry")
    assert dom == "historical_analogy"


def test_infer_security_domain_keywords():
    dom = rag._infer_domain("notes.txt", "Taiwan deterrence escalation crisis")
    assert dom == "security_taiwan"


def test_retrieve_candidates_and_filters(kb_with_samples):
    clear_retrieval_cache()
    q = "Taiwan deterrence Cold War"
    all_cands = retrieve_candidates(q, candidate_k=20, chunks_path=kb_with_samples)
    assert len(all_cands) > 0

    filt = RetrievalFilters(domains=["security_taiwan", "general"])
    sec_cands = retrieve_candidates(
        q, filters=filt, candidate_k=20, chunks_path=kb_with_samples
    )
    assert len(sec_cands) > 0
    for ch in sec_cands:
        assert ch.domain in ("security_taiwan", "general")


def test_rerank_respects_final_k(kb_with_samples):
    cands = retrieve_candidates("chip trade", candidate_k=15, chunks_path=kb_with_samples)
    finals = rerank_candidates("chip trade", cands, 3)
    assert len(finals) <= 3


def test_retrieve_for_agent_profile(kb_with_samples):
    clear_retrieval_cache()
    metrics = RunMetrics()
    rec = RagMetricsRecorder(metrics)
    chunks = retrieve_for_agent(
        "China blockades Taiwan",
        "escalation",
        "historical_analogy",
        round_number=1,
        recorder=rec,
        chunks_path=kb_with_samples,
    )
    assert len(chunks) <= ACTIVE_RAG_CONFIG["agent_round1_final_k"]
    assert metrics.rag_calls >= 1


def test_retrieve_for_disagreement_uses_terms(kb_with_samples):
    clear_retrieval_cache()
    summary = DiscussionSummary(
        round_number=1,
        areas_of_disagreement=["Pace of decoupling vs re-engagement"],
        key_uncertainties=["Taiwan crisis timing"],
        disagreement_query_terms=["decoupling", "taiwan"],
    )
    chunks = retrieve_for_disagreement(
        "seed about Taiwan",
        "base_case",
        summary,
        chunks_path=kb_with_samples,
    )
    assert len(chunks) <= ACTIVE_RAG_CONFIG["disagreement_final_k"]


def test_retrieve_for_red_team(kb_with_samples):
    clear_retrieval_cache()
    chunks = retrieve_for_red_team(
        "seed", "base_case", None, chunks_path=kb_with_samples
    )
    assert len(chunks) <= ACTIVE_RAG_CONFIG["red_team_final_k"]


def test_evidence_lanes_and_limits(kb_with_samples):
    baseline = retrieve_baseline(
        "US China rivalry", "base_case", chunks_path=kb_with_samples
    )
    summary = EvidenceSummary(
        observed_facts=["Fact A"],
        historical_analogies=["Analogy B"],
        strategy_frameworks=["Framework C"],
    )
    lanes = build_evidence_lanes(baseline, summary)
    assert isinstance(lanes, EvidenceLanes)
    assert lanes.historical_blob or lanes.frameworks_blob or lanes.general_blob
    max_lane = ACTIVE_RAG_CONFIG["max_lane_chars"]
    for field in EvidenceLanes.model_fields:
        val = getattr(lanes, field)
        assert len(val) <= max_lane + 50


def test_empty_kb_lanes_valid(tmp_path):
    out = tmp_path / "empty.json"
    rag.ingest_knowledge_base(str(tmp_path / "missing_kb"), str(out))
    lanes = build_evidence_lanes([], None)
    assert lanes.observed_blob == ""


def test_citation_validation():
    data = {
        "sources_used": ["kb_000001", "kb_999999"],
        "grounding_notes": [
            {"chunk_id": "kb_000001", "claim": "valid"},
            {"chunk_id": "kb_bad", "claim": "invalid"},
        ],
        "rag_influence": "supported_view",
        "rag_influence_explanation": "ok",
    }
    metrics = RunMetrics()
    rec = RagMetricsRecorder(metrics)
    cleaned, warnings = validate_and_apply_citations(
        data, ["kb_000001"], "geo_strategy", rec
    )
    assert cleaned["sources_used"] == ["kb_000001"]
    assert len(cleaned["grounding_notes"]) == 1
    assert warnings
    assert metrics.citation_warnings


def test_build_agent_packet_round1_vs_round2():
    lanes = EvidenceLanes(observed_blob="Observed lane text")
    agent_chunks = [
        EvidenceChunk(chunk_id="kb_000001", text="Agent chunk", domain="geo_strategy")
    ]
    dispute = [
        EvidenceChunk(chunk_id="kb_000002", text="Dispute chunk", domain="general")
    ]
    p1, ids1 = build_agent_evidence_packet(
        "geo_strategy", lanes, agent_chunks, round_number=1
    )
    p2, ids2 = build_agent_evidence_packet(
        "geo_strategy", lanes, [], disagreement_chunks=dispute, round_number=2
    )
    assert "kb_000001" in ids1
    assert "kb_000002" in ids2
    assert "Dispute" in p2 or "kb_000002" in p2
    assert "Dispute" not in p1 or "kb_000002" not in p1


def test_final_evidence_packet_cap():
    chunks = [
        EvidenceChunk(chunk_id=f"kb_{i:06d}", source_path=f"f{i}.md", text=f"t{i}")
        for i in range(20)
    ]
    outputs = {
        "geo_strategy": [
            AgentOutput(
                agent_name="geo_strategy",
                sources_used=["kb_000001", "kb_000002"],
            )
        ]
    }
    packet = build_final_evidence_packet(
        baseline_chunks=chunks,
        disagreement_chunks=[],
        red_team_chunks=[],
        lanes=EvidenceLanes(),
        agent_outputs=outputs,
        max_items=5,
    )
    assert len(packet.items) <= 5


def test_retrieval_cache_stable_key(kb_with_samples):
    clear_retrieval_cache()
    cache = {}
    c1, h1 = rag.retrieve_cached(
        seed="s",
        scenario_mode="base_case",
        agent_name="geo_strategy",
        round_number=1,
        query="US China alliances",
        filters=RetrievalFilters(domains=["geo_strategy"]),
        candidate_k=10,
        final_k=3,
        cache=cache,
        chunks_path=kb_with_samples,
    )
    c2, h2 = rag.retrieve_cached(
        seed="s",
        scenario_mode="base_case",
        agent_name="geo_strategy",
        round_number=1,
        query="US China alliances",
        filters=RetrievalFilters(domains=["geo_strategy"]),
        candidate_k=10,
        final_k=3,
        cache=cache,
        chunks_path=kb_with_samples,
    )
    assert h1 is False
    assert h2 is True
    assert [x.chunk_id for x in c1] == [x.chunk_id for x in c2]


def test_graph_tier2_flow_mock_mode():
    node_names = [n[0] for n in NODES]
    assert "year_2026_cycle" in node_names
    assert "year_2031_cycle" in node_names

    final = run_graph(seed="Taiwan crisis and Cold War analogy", scenario_mode="base_case")
    assert final.run_metrics.rag_calls >= 1
    assert final.run_metrics.years_completed == 6
    assert "evidence_rag" in final.run_metrics.agents_used


def test_load_chunks_reinfers_domain_from_text(tmp_path):
    out = tmp_path / "chunks.json"
    import json

    with open(out, "w", encoding="utf-8") as fh:
        json.dump(
            [
                {
                    "chunk_id": "kb_000001",
                    "source_path": "book.pdf",
                    "domain": "general",
                    "text": "Taiwan deterrence and crisis escalation in the strait.",
                }
            ],
            fh,
        )
    loaded = rag.load_chunks(str(out))
    assert loaded[0]["domain"] == "security_taiwan"


def test_round3_no_extra_agent_retrieval(monkeypatch, kb_with_samples):
    """Round 1 agent retrieval only; round 3 should not call retrieve_for_agent."""
    calls = []

    def _spy(seed, scenario_mode, agent_name, target_year=2026, round_number=1, recorder=None):
        calls.append((agent_name, target_year, round_number))
        return []

    monkeypatch.setattr(rag, "retrieve_for_agent", _spy)
    from app.llm import LLMClient
    from app.schemas import EvidenceLanes, RunMetrics, ScenarioState
    from app.year_simulation import _run_discussion_round

    state = ScenarioState(run_id="t", seed="x", scenario_mode="base_case", current_year=2027)
    state.evidence_lanes = EvidenceLanes()
    state.run_metrics = RunMetrics()
    recorder = rag.RagMetricsRecorder(metrics=state.run_metrics)
    _run_discussion_round(state, LLMClient(), 3, recorder)
    assert calls == []
