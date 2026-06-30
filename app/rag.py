"""Local RAG: ingestion, metadata, Tier 2 retrieval, evidence lanes."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from . import config as _config_mod
from .rag_config import ACTIVE_RAG_CONFIG, load_rag_config
from .schemas import (
    DiscussionSummary,
    EvidenceChunk,
    EvidenceLanes,
    EvidenceSummary,
    FinalEvidencePacket,
    RunMetrics,
)
from .utils import stable_hash, truncate
from . import rag_vector_store as vector_store


SUPPORTED_EXTENSIONS = (".txt", ".pdf")
DEFAULT_CHUNK_CHARS = 1200
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_PREPROCESSED_DIR = "data/preprocessed"

VALID_DOMAINS = frozenset(
    {
        "economy_technology",
        "security_taiwan",
        "geo_strategy",
        "domestic_ideology",
        "historical_analogy",
        "strategy_framework",
        "general",
        "unknown",
    }
)

DOMAIN_KEYWORD_RULES: List[Tuple[str, Tuple[str, ...]]] = [
    ("security_taiwan", ("taiwan", "deterrence", "escalation", "crisis", "military", "gray-zone", "gray zone")),
    ("economy_technology", ("chip", "semiconductor", "trade", "sanction", "economy", "supply chain", "export control", "rare earth")),
    ("domestic_ideology", ("ideology", "nationalism", "legitimacy", "domestic", "propaganda", "ccp", "regime")),
    ("geo_strategy", ("alliance", "indo-pacific", "indo pacific", "diplomacy", "grand strategy", "balance of power")),
    ("historical_analogy", ("cold_war", "cold war", "ussr", "historical", "analogy", "soviet")),
    ("strategy_framework", ("framework", "scenario", "red_team", "red team", "escalation_ladder", "escalation ladder", "containment", "doctrine")),
]

AGENT_RETRIEVAL_PROFILES: Dict[str, Dict[str, Any]] = {
    "geo_strategy": {
        "domains": ["geo_strategy", "strategy_framework", "general"],
        "query_suffix": (
            "alliances balance of power Indo-Pacific grand strategy diplomacy "
            "containment influence competition"
        ),
    },
    "economy_technology": {
        "domains": ["economy_technology", "general"],
        "query_suffix": (
            "trade semiconductors AI chips export controls sanctions rare earths "
            "supply chains interdependence financial stress industrial policy"
        ),
    },
    "domestic_ideology": {
        "domains": ["domestic_ideology", "general"],
        "query_suffix": (
            "domestic politics nationalism ideology legitimacy public opinion "
            "propaganda regime stability elite incentives"
        ),
    },
    "security_taiwan": {
        "domains": ["security_taiwan", "strategy_framework", "general"],
        "query_suffix": (
            "Taiwan deterrence crisis stability gray-zone pressure escalation "
            "de-escalation accidental escalation strategic signaling"
        ),
    },
    "historical_analogy": {
        "domains": ["historical_analogy", "strategy_framework", "general"],
        "query_suffix": (
            "US USSR Cold War analogy containment crisis management deterrence "
            "proxy competition arms race failed analogy"
        ),
    },
    "red_team": {
        "domains": ["strategy_framework", "historical_analogy", "general"],
        "query_suffix": (
            "overconfidence omitted variable failed analogy contradiction uncertainty "
            "economic interdependence crisis miscalculation deterrence failure"
        ),
    },
}


@dataclass
class IngestionResult:
    chunk_count: int
    files_processed: int
    output_path: str
    vector_index_path: str = ""
    pdf_files: int = 0
    text_files: int = 0
    skipped_files: int = 0


@dataclass
class RetrievalFilters:
    domains: Optional[List[str]] = None
    source_types: Optional[List[str]] = None
    periods: Optional[List[str]] = None


@dataclass
class RagMetricsRecorder:
    """Accumulate RAG stats into RunMetrics during a simulation run."""

    metrics: RunMetrics
    _all_chunk_ids: Set[str] = field(default_factory=set)
    _source_counts: Dict[str, int] = field(default_factory=dict)
    _citation_counts: Dict[str, int] = field(default_factory=dict)

    def record_retrieval(
        self,
        *,
        cache_hit: bool,
        candidate_count: int,
        final_chunks: List[EvidenceChunk],
    ) -> None:
        if cache_hit:
            self.metrics.rag_cache_hits += 1
        else:
            self.metrics.rag_calls += 1
        self.metrics.retrieved_candidate_chunks += candidate_count
        self.metrics.retrieved_final_chunks += len(final_chunks)
        for ch in final_chunks:
            if ch.chunk_id:
                self._all_chunk_ids.add(ch.chunk_id)
            if ch.source_path:
                self._source_counts[ch.source_path] = (
                    self._source_counts.get(ch.source_path, 0) + 1
                )
        self._sync()

    def record_citations(self, agent_name: str, chunk_ids: List[str]) -> None:
        if chunk_ids:
            self.metrics.per_agent_sources_used[agent_name] = list(chunk_ids)
        for cid in chunk_ids:
            self._citation_counts[cid] = self._citation_counts.get(cid, 0) + 1

    def add_warning(self, msg: str) -> None:
        if msg and msg not in self.metrics.citation_warnings:
            self.metrics.citation_warnings.append(msg[:300])

    def _sync(self) -> None:
        self.metrics.unique_chunks_used = len(self._all_chunk_ids)
        self.metrics.retrieved_chunk_ids = sorted(self._all_chunk_ids)
        self.metrics.rag_source_files = sorted(self._source_counts.keys())
        ranked = sorted(self._source_counts.items(), key=lambda x: (-x[1], x[0]))
        self.metrics.most_used_source_files = [p for p, _ in ranked[:8]]
        cited = sorted(self._citation_counts.items(), key=lambda x: (-x[1], x[0]))
        self.metrics.most_cited_chunk_ids = [c for c, _ in cited[:12]]
        self.metrics.retrieved_docs = self.metrics.retrieved_final_chunks


def _infer_source_name(path: str) -> str:
    return os.path.basename(path) or "unknown"


def _infer_source_type(path: str) -> str:
    lowered = path.lower()
    if any(k in lowered for k in ("book", "books/")):
        return "book"
    if any(k in lowered for k in ("report", "brief", "paper")):
        return "report"
    if "framework" in lowered:
        return "framework"
    if any(k in lowered for k in ("current", "context", "news", "2024", "2025")):
        return "current_context"
    if any(k in lowered for k in ("history", "historical", "cold war", "ussr", "analogy")):
        return "historical_analogy"
    if any(k in lowered for k in ("strategy", "doctrine", "containment", "framework")):
        return "strategy_framework"
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return "book"
    return "unknown"


def _infer_period(path: str, text: str = "") -> str:
    blob = (path + " " + text[:500]).lower()
    if any(k in blob for k in ("cold war", "ussr", "soviet", "berlin", "nato 194", "kennan")):
        return "us_ussr_cold_war"
    if any(k in blob for k in ("china", "taiwan", "sino", "ccp", "indo-pacific", "semiconductor")):
        return "modern_us_china"
    return "general"


def _infer_domain(path: str, text: str = "") -> str:
    blob = (path + " " + text[:400]).lower().replace("-", "_").replace(" ", "_")
    scores: Dict[str, int] = {}
    for domain, keywords in DOMAIN_KEYWORD_RULES:
        scores[domain] = sum(1 for kw in keywords if kw.replace(" ", "_") in blob or kw in blob)
    best = max(scores.items(), key=lambda x: x[1])
    if best[1] > 0:
        return best[0]
    parts = re.split(r"[\\/]", path.lower())
    folder_map = {
        "economy": "economy_technology",
        "technology": "economy_technology",
        "security": "security_taiwan",
        "strategy": "geo_strategy",
        "ideology": "domestic_ideology",
        "historical": "historical_analogy",
        "history": "historical_analogy",
        "framework": "strategy_framework",
        "frameworks": "strategy_framework",
    }
    for p in parts:
        if p in folder_map:
            return folder_map[p]
    return "general"


def _normalize_chunk_dict(raw: Dict[str, Any], index: int = 0) -> Dict[str, Any]:
    text = str(raw.get("text") or "")
    path = str(raw.get("source_path") or "")
    chunk_id = str(raw.get("chunk_id") or raw.get("id") or f"kb_{index:06d}")
    domain = raw.get("domain")
    if domain in ("economy", "security", "historical", "ideology", "strategy"):
        domain = {
            "economy": "economy_technology",
            "security": "security_taiwan",
            "historical": "historical_analogy",
            "ideology": "domestic_ideology",
            "strategy": "geo_strategy",
        }.get(domain, domain)
    if not domain or domain in ("general", "unknown") or domain not in VALID_DOMAINS:
        domain = _infer_domain(path, text)
    return {
        "chunk_id": chunk_id,
        "source_path": path,
        "source_name": raw.get("source_name") or _infer_source_name(path),
        "domain": domain,
        "source_type": raw.get("source_type") or _infer_source_type(path),
        "period": raw.get("period") or _infer_period(path, text),
        "text": text,
        "char_count": len(text),
    }


def load_chunks(path: Optional[str] = None) -> List[Dict[str, Any]]:
    path = path or _config_mod.CONFIG.rag_chunks_path
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            return []
    except Exception:
        return []
    return [_normalize_chunk_dict(c, i) for i, c in enumerate(data)]


def _dict_to_evidence_chunk(ch: Dict[str, Any], score: float = 0.0) -> EvidenceChunk:
    max_chars = int(ACTIVE_RAG_CONFIG.get("max_chunk_chars", 900))
    text = truncate(str(ch.get("text") or ""), max_chars)
    return EvidenceChunk(
        chunk_id=str(ch.get("chunk_id") or ""),
        source_path=str(ch.get("source_path") or ""),
        source_name=str(ch.get("source_name") or ""),
        domain=str(ch.get("domain") or "general"),
        source_type=str(ch.get("source_type") or "unknown"),
        period=str(ch.get("period") or "unknown"),
        text=text,
        score=float(score),
        char_count=int(ch.get("char_count") or len(text)),
    )


_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _keyword_score(query_tokens: List[str], doc_text: str) -> float:
    if not query_tokens:
        return 0.0
    doc_tokens = _tokenize(doc_text)
    if not doc_tokens:
        return 0.0
    qset = set(query_tokens)
    dset = set(doc_tokens)
    overlap = len(qset & dset)
    if overlap == 0:
        return 0.0
    return overlap / (len(qset) + 1e-9)


def _tfidf_scores(query: str, docs: List[str]) -> List[float]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception:
        return []
    if not docs:
        return []
    vec = TfidfVectorizer(stop_words="english", lowercase=True)
    matrix = vec.fit_transform(docs + [query])
    sims = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    return [float(s) for s in sims]


def _metadata_bonus(
    chunk: Dict[str, Any],
    filters: Optional[RetrievalFilters],
    query_tokens: List[str],
) -> float:
    bonus = 0.0
    if filters and filters.domains:
        dom = chunk.get("domain") or "general"
        if dom in filters.domains or dom == "general":
            bonus += 0.08
        else:
            bonus -= 0.15
    if filters and filters.source_types:
        st = chunk.get("source_type") or "unknown"
        if st in filters.source_types:
            bonus += 0.05
    if filters and filters.periods:
        per = chunk.get("period") or "unknown"
        if per in filters.periods or per == "general":
            bonus += 0.04
    path = (chunk.get("source_path") or "").lower()
    for tok in query_tokens[:12]:
        if len(tok) > 3 and tok in path:
            bonus += 0.02
    return bonus


def _passes_filters(chunk: Dict[str, Any], filters: Optional[RetrievalFilters]) -> bool:
    if not filters:
        return True
    dom = chunk.get("domain") or "general"
    if filters.domains and dom not in filters.domains and dom != "general":
        return False
    st = chunk.get("source_type") or "unknown"
    if filters.source_types and st not in filters.source_types:
        return False
    per = chunk.get("period") or "unknown"
    if filters.periods and per not in filters.periods and per != "general":
        return False
    return True


def _filters_to_chroma_where(filters: Optional[RetrievalFilters]) -> Optional[Dict[str, Any]]:
    if not filters:
        return None
    clauses: List[Dict[str, Any]] = []
    if filters.domains:
        domains = list(dict.fromkeys(list(filters.domains) + ["general"]))
        clauses.append({"domain": {"$in": domains}})
    if filters.source_types:
        clauses.append({"source_type": {"$in": list(filters.source_types)}})
    if filters.periods:
        periods = list(dict.fromkeys(list(filters.periods) + ["general"]))
        clauses.append({"period": {"$in": periods}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _retrieve_candidates_tfidf(
    query: str,
    filters: Optional[RetrievalFilters],
    candidate_k: int,
    chunks_path: Optional[str],
) -> List[EvidenceChunk]:
    chunks = load_chunks(chunks_path)
    if not chunks:
        return []

    filtered = [c for c in chunks if _passes_filters(c, filters)]
    if not filtered:
        filtered = chunks

    docs = [c.get("text", "") for c in filtered]
    scores = _tfidf_scores(query, docs)
    q_tokens = _tokenize(query)
    if not scores:
        scores = [_keyword_score(q_tokens, d) for d in docs]

    ranked: List[Tuple[float, Dict[str, Any]]] = []
    for score, ch in zip(scores, filtered):
        total = float(score) + _metadata_bonus(ch, filters, q_tokens)
        if total > 0:
            ranked.append((total, ch))
    ranked.sort(key=lambda x: x[0], reverse=True)

    out: List[EvidenceChunk] = []
    for score, ch in ranked[:candidate_k]:
        out.append(_dict_to_evidence_chunk(ch, score))
    return out


def _retrieve_candidates_chroma(
    query: str,
    filters: Optional[RetrievalFilters],
    candidate_k: int,
    chunks_path: Optional[str],
    chroma_path: Optional[str],
) -> List[EvidenceChunk]:
    where = _filters_to_chroma_where(filters)
    hits = vector_store.query_chunks(
        query,
        n_results=candidate_k,
        where=where,
        chunks_path=chunks_path,
        chroma_path=chroma_path,
    )
    if not hits and where is not None:
        hits = vector_store.query_chunks(
            query,
            n_results=candidate_k,
            where=None,
            chunks_path=chunks_path,
            chroma_path=chroma_path,
        )

    q_tokens = _tokenize(query)
    out: List[EvidenceChunk] = []
    for ch, score in hits:
        if filters and not _passes_filters(ch, filters):
            continue
        total = float(score) + _metadata_bonus(ch, filters, q_tokens)
        if total <= 0:
            continue
        out.append(_dict_to_evidence_chunk(ch, total))
    out.sort(key=lambda c: c.score, reverse=True)
    return out[:candidate_k]


def retrieve_candidates(
    query: str,
    filters: Optional[RetrievalFilters] = None,
    candidate_k: Optional[int] = None,
    chunks_path: Optional[str] = None,
    chroma_path: Optional[str] = None,
) -> List[EvidenceChunk]:
    cfg = ACTIVE_RAG_CONFIG
    candidate_k = candidate_k or int(cfg.get("baseline_candidate_k", 25))

    indexed = vector_store.collection_count(
        chunks_path=chunks_path,
        chroma_path=chroma_path,
    )
    if indexed > 0:
        return _retrieve_candidates_chroma(
            query,
            filters,
            candidate_k,
            chunks_path,
            chroma_path,
        )
    return _retrieve_candidates_tfidf(query, filters, candidate_k, chunks_path)


def rerank_candidates(
    query: str,
    candidates: List[EvidenceChunk],
    final_k: int,
) -> List[EvidenceChunk]:
    if not candidates:
        return []
    if len(candidates) <= final_k:
        return candidates

    max_per_source = int(ACTIVE_RAG_CONFIG.get("max_chunks_per_source", 2))
    q_tokens = _tokenize(query)
    selected: List[EvidenceChunk] = []
    per_source: Dict[str, int] = {}

    for ch in sorted(candidates, key=lambda c: c.score, reverse=True):
        src = ch.source_path or ch.source_name or "unknown"
        if per_source.get(src, 0) >= max_per_source:
            continue
        overlap = _keyword_score(q_tokens, ch.text)
        ch.score = ch.score * 0.7 + overlap * 0.3
        selected.append(ch)
        per_source[src] = per_source.get(src, 0) + 1
        if len(selected) >= final_k:
            break

    if len(selected) < final_k:
        seen = {c.chunk_id for c in selected}
        for ch in sorted(candidates, key=lambda c: c.score, reverse=True):
            if ch.chunk_id in seen:
                continue
            selected.append(ch)
            seen.add(ch.chunk_id)
            if len(selected) >= final_k:
                break
    return selected[:final_k]


def _build_query(seed: str, scenario_mode: str, suffix: str = "") -> str:
    parts = [seed, scenario_mode]
    if suffix:
        parts.append(suffix)
    return " ".join(p for p in parts if p)


_RETRIEVAL_CACHE: Dict[str, List[EvidenceChunk]] = {}


def _cache_key(
    seed: str,
    scenario_mode: str,
    agent_name: str,
    round_number: int,
    query: str,
    filters: Optional[RetrievalFilters],
) -> str:
    filt = ""
    if filters:
        filt = json.dumps(
            {
                "d": filters.domains,
                "s": filters.source_types,
                "p": filters.periods,
            },
            sort_keys=True,
        )
    return stable_hash(seed, scenario_mode, agent_name, round_number, query, filt)


def retrieve_cached(
    seed: str,
    scenario_mode: str,
    agent_name: str,
    round_number: int,
    query: str,
    filters: Optional[RetrievalFilters],
    candidate_k: int,
    final_k: int,
    recorder: Optional[RagMetricsRecorder] = None,
    cache: Optional[Dict[str, List[EvidenceChunk]]] = None,
    chunks_path: Optional[str] = None,
) -> Tuple[List[EvidenceChunk], bool]:
    if cache is None:
        cache = _RETRIEVAL_CACHE
    key = _cache_key(seed, scenario_mode, agent_name, round_number, query, filters)
    if key in cache:
        chunks = cache[key]
        if recorder:
            recorder.record_retrieval(
                cache_hit=True, candidate_count=len(chunks), final_chunks=chunks
            )
        return chunks, True

    candidates = retrieve_candidates(
        query, filters=filters, candidate_k=candidate_k, chunks_path=chunks_path
    )
    finals = rerank_candidates(query, candidates, final_k)
    cache[key] = finals
    if recorder:
        recorder.record_retrieval(
            cache_hit=False, candidate_count=len(candidates), final_chunks=finals
        )
    return finals, False


def retrieve_baseline(
    seed: str,
    scenario_mode: str,
    recorder: Optional[RagMetricsRecorder] = None,
    chunks_path: Optional[str] = None,
) -> List[EvidenceChunk]:
    cfg = ACTIVE_RAG_CONFIG
    query = _build_query(
        seed,
        scenario_mode,
        "US China rivalry Cold War strategy frameworks historical analogy",
    )
    if not _config_mod.CONFIG.use_rag:
        return []
    candidates = retrieve_candidates(
        query,
        candidate_k=int(cfg.get("baseline_candidate_k", 25)),
        chunks_path=chunks_path,
    )
    finals = rerank_candidates(
        query, candidates, int(cfg.get("baseline_final_k", 8))
    )
    if recorder:
        recorder.record_retrieval(
            cache_hit=False,
            candidate_count=len(candidates),
            final_chunks=finals,
        )
    return finals


def retrieve_for_agent(
    seed: str,
    scenario_mode: str,
    agent_name: str,
    target_year: int = 2026,
    round_number: int = 1,
    recorder: Optional[RagMetricsRecorder] = None,
    chunks_path: Optional[str] = None,
) -> List[EvidenceChunk]:
    cfg = ACTIVE_RAG_CONFIG
    if not _config_mod.CONFIG.use_rag or not cfg.get("enable_agent_rag", True):
        return []
    if round_number != 1:
        return []

    profile = AGENT_RETRIEVAL_PROFILES.get(agent_name)
    if not profile:
        return []

    query = _build_query(
        seed,
        scenario_mode,
        "year {y} ".format(y=target_year) + profile.get("query_suffix", ""),
    )
    filters = RetrievalFilters(domains=list(profile.get("domains") or []))
    return retrieve_cached(
        seed=seed,
        scenario_mode=scenario_mode,
        agent_name=agent_name,
        round_number=round_number,
        query=query,
        filters=filters,
        candidate_k=int(cfg.get("agent_round1_candidate_k", 15)),
        final_k=int(cfg.get("agent_round1_final_k", 4)),
        recorder=recorder,
        chunks_path=chunks_path,
    )[0]


def retrieve_for_disagreement(
    seed: str,
    scenario_mode: str,
    discussion_summary: Optional[DiscussionSummary],
    target_year: int = 2026,
    recorder: Optional[RagMetricsRecorder] = None,
    chunks_path: Optional[str] = None,
) -> List[EvidenceChunk]:
    cfg = ACTIVE_RAG_CONFIG
    if not _config_mod.CONFIG.use_rag or not cfg.get("enable_disagreement_rag", True):
        return []
    if discussion_summary is None:
        return []

    terms: List[str] = []
    terms.extend(discussion_summary.areas_of_disagreement[:4])
    terms.extend(discussion_summary.key_uncertainties[:3])
    terms.extend(discussion_summary.disagreement_query_terms[:4])
    if not terms:
        return []

    query = _build_query(
        seed,
        scenario_mode,
        "year {y} ".format(y=target_year) + " ".join(terms),
    )
    filters = RetrievalFilters(
        domains=["strategy_framework", "historical_analogy", "general", "security_taiwan"]
    )
    return retrieve_cached(
        seed=seed,
        scenario_mode=scenario_mode,
        agent_name="disagreement",
        round_number=2,
        query=query,
        filters=filters,
        candidate_k=int(cfg.get("disagreement_candidate_k", 15)),
        final_k=int(cfg.get("disagreement_final_k", 4)),
        recorder=recorder,
        chunks_path=chunks_path,
    )[0]


def retrieve_for_red_team(
    seed: str,
    scenario_mode: str,
    final_discussion_summary: Optional[DiscussionSummary],
    recorder: Optional[RagMetricsRecorder] = None,
    chunks_path: Optional[str] = None,
) -> List[EvidenceChunk]:
    cfg = ACTIVE_RAG_CONFIG
    if not _config_mod.CONFIG.use_rag or not cfg.get("enable_red_team_rag", True):
        return []

    summary_bits: List[str] = []
    if final_discussion_summary:
        summary_bits.extend(final_discussion_summary.areas_of_disagreement[:3])
        summary_bits.extend(final_discussion_summary.key_uncertainties[:2])

    profile = AGENT_RETRIEVAL_PROFILES["red_team"]
    query = _build_query(
        seed,
        scenario_mode,
        profile.get("query_suffix", "") + " " + " ".join(summary_bits),
    )
    filters = RetrievalFilters(domains=list(profile.get("domains") or []))
    return retrieve_cached(
        seed=seed,
        scenario_mode=scenario_mode,
        agent_name="red_team",
        round_number=99,
        query=query,
        filters=filters,
        candidate_k=int(cfg.get("red_team_candidate_k", 15)),
        final_k=int(cfg.get("red_team_final_k", 4)),
        recorder=recorder,
        chunks_path=chunks_path,
    )[0]


def _append_lane(blob: str, piece: str, max_chars: int) -> str:
    if not piece:
        return blob
    sep = "\n" if blob else ""
    return truncate(blob + sep + piece, max_chars)


def build_evidence_lanes(
    chunks: List[EvidenceChunk],
    evidence_summary: Optional[EvidenceSummary] = None,
) -> EvidenceLanes:
    max_lane = int(ACTIVE_RAG_CONFIG.get("max_lane_chars", 1200))
    lanes = EvidenceLanes()

    for ch in chunks:
        line = "[{id}] {src}: {txt}".format(
            id=ch.chunk_id,
            src=ch.source_name or ch.source_path,
            txt=truncate(ch.text, 280),
        )
        dom = ch.domain or "general"
        st = ch.source_type or "unknown"
        if st == "current_context" or dom == "economy_technology" and "current" in (ch.source_path or "").lower():
            lanes.observed_blob = _append_lane(lanes.observed_blob, line, max_lane)
        if dom == "historical_analogy" or st == "historical_analogy":
            lanes.historical_blob = _append_lane(lanes.historical_blob, line, max_lane)
        if dom == "strategy_framework" or st in ("strategy_framework", "framework"):
            lanes.frameworks_blob = _append_lane(lanes.frameworks_blob, line, max_lane)
        if dom == "economy_technology":
            lanes.economy_blob = _append_lane(lanes.economy_blob, line, max_lane)
        if dom == "security_taiwan":
            lanes.security_blob = _append_lane(lanes.security_blob, line, max_lane)
        if dom == "domestic_ideology":
            lanes.domestic_blob = _append_lane(lanes.domestic_blob, line, max_lane)
        if dom == "geo_strategy":
            lanes.geostrategy_blob = _append_lane(lanes.geostrategy_blob, line, max_lane)
        lanes.general_blob = _append_lane(lanes.general_blob, line, max_lane)

    if evidence_summary:
        if evidence_summary.observed_facts:
            lanes.observed_blob = _append_lane(
                lanes.observed_blob,
                "Observed: " + "; ".join(evidence_summary.observed_facts[:5]),
                max_lane,
            )
        if evidence_summary.historical_analogies:
            lanes.historical_blob = _append_lane(
                lanes.historical_blob,
                "Analogies: " + "; ".join(evidence_summary.historical_analogies[:5]),
                max_lane,
            )
        if evidence_summary.strategy_frameworks:
            lanes.frameworks_blob = _append_lane(
                lanes.frameworks_blob,
                "Frameworks: " + "; ".join(evidence_summary.strategy_frameworks[:5]),
                max_lane,
            )

    return lanes


def build_final_evidence_packet(
    *,
    baseline_chunks: List[EvidenceChunk],
    disagreement_chunks: List[EvidenceChunk],
    red_team_chunks: List[EvidenceChunk],
    lanes: EvidenceLanes,
    agent_outputs: Dict[str, List[Any]],
    max_items: Optional[int] = None,
) -> FinalEvidencePacket:
    max_items = max_items or int(ACTIVE_RAG_CONFIG.get("max_final_evidence_items", 12))
    cited_counts: Dict[str, int] = {}
    for outputs in agent_outputs.values():
        for out in outputs:
            for cid in getattr(out, "sources_used", None) or []:
                if cid:
                    cited_counts[cid] = cited_counts.get(cid, 0) + 1

    registry: Dict[str, EvidenceChunk] = {}
    for ch in baseline_chunks + disagreement_chunks + red_team_chunks:
        if ch.chunk_id:
            registry[ch.chunk_id] = ch

    items: List[str] = []
    chunk_ids: List[str] = []
    source_files: Set[str] = set()

    def _add_item(label: str, ch: EvidenceChunk) -> None:
        if len(items) >= max_items:
            return
        items.append(
            "{label} [{id}] {src}: {txt}".format(
                label=label,
                id=ch.chunk_id,
                src=ch.source_name or ch.source_path,
                txt=truncate(ch.text, 220),
            )
        )
        if ch.chunk_id and ch.chunk_id not in chunk_ids:
            chunk_ids.append(ch.chunk_id)
        if ch.source_path:
            source_files.add(ch.source_path)

    ranked_cited = sorted(cited_counts.items(), key=lambda x: (-x[1], x[0]))
    for cid, _ in ranked_cited:
        ch = registry.get(cid)
        if ch:
            _add_item("Cited", ch)

    for label, chunk_list in (
        ("Observed", [c for c in baseline_chunks if c.source_type == "current_context"]),
        ("Historical", [c for c in baseline_chunks if c.domain == "historical_analogy"]),
        ("Framework", [c for c in baseline_chunks if c.domain == "strategy_framework"]),
        ("Dispute", disagreement_chunks),
        ("Red-team", red_team_chunks),
    ):
        for ch in chunk_list:
            _add_item(label, ch)

    if len(items) < max_items and lanes.historical_blob:
        items.append("Lane historical: " + truncate(lanes.historical_blob, 300))
    if len(items) < max_items and lanes.frameworks_blob:
        items.append("Lane frameworks: " + truncate(lanes.frameworks_blob, 300))

    text = "\n".join(items[:max_items])
    return FinalEvidencePacket(
        items=items[:max_items],
        source_files=sorted(source_files),
        chunk_ids=chunk_ids,
        text=text,
    )


# --- Ingestion (PDF / md / txt) --------------------------------------------


def _preprocessed_cache_path(source_path: str, preprocessed_dir: str) -> str:
    base = os.path.basename(source_path)
    stem, _ext = os.path.splitext(base)
    safe = re.sub(r"[^\w\-.]+", "_", stem).strip("_") or "document"
    return os.path.join(preprocessed_dir, safe + ".txt")


def _extract_pdf_text(path: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise RuntimeError(
            "PDF support requires the 'pypdf' package. Run: pip install pypdf"
        ) from e

    reader = PdfReader(path)
    parts: List[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = text.strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def _normalize_extracted_text(text: str) -> str:
    text = (text or "").replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _read_source_text(
    path: str,
    preprocessed_dir: Optional[str] = None,
) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            return _normalize_extracted_text(fh.read())

    if ext == ".pdf":
        pre_dir = preprocessed_dir or DEFAULT_PREPROCESSED_DIR
        cache_path = _preprocessed_cache_path(path, pre_dir)
        pdf_mtime = os.path.getmtime(path)
        if os.path.exists(cache_path):
            try:
                if os.path.getmtime(cache_path) >= pdf_mtime:
                    with open(cache_path, "r", encoding="utf-8", errors="ignore") as fh:
                        cached = _normalize_extracted_text(fh.read())
                    if cached:
                        return cached
            except OSError:
                pass

        raw = _normalize_extracted_text(_extract_pdf_text(path))
        parent = os.path.dirname(cache_path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as fh:
            fh.write(raw)
        return raw

    return ""


def _iter_source_files(root: str) -> Iterable[str]:
    if not os.path.isdir(root):
        return
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if f.startswith(".") or f.endswith("/.gitkeep"):
                continue
            if not f.lower().endswith(SUPPORTED_EXTENSIONS):
                continue
            yield os.path.join(dirpath, f)


def _chunk_text(
    text: str,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_chars:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_chars, len(text))
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(0, end - overlap)
    return [c for c in chunks if c]


def ingest_knowledge_base(
    kb_dir: str = "knowledge_base",
    output_path: Optional[str] = None,
    preprocessed_dir: Optional[str] = None,
) -> IngestionResult:
    output_path = output_path or _config_mod.CONFIG.rag_chunks_path
    preprocessed_dir = preprocessed_dir or DEFAULT_PREPROCESSED_DIR
    stored: List[Dict[str, Any]] = []
    files_processed = 0
    pdf_files = 0
    text_files = 0
    skipped_files = 0
    seq = 0

    for path in _iter_source_files(kb_dir):
        ext = os.path.splitext(path)[1].lower()
        try:
            raw = _read_source_text(path, preprocessed_dir=preprocessed_dir)
        except Exception:
            skipped_files += 1
            continue
        if not raw:
            skipped_files += 1
            continue

        files_processed += 1
        if ext == ".pdf":
            pdf_files += 1
        else:
            text_files += 1

        for piece in _chunk_text(raw):
            seq += 1
            meta = _normalize_chunk_dict(
                {
                    "chunk_id": f"kb_{seq:06d}",
                    "source_path": path,
                    "text": piece,
                },
                seq,
            )
            stored.append(meta)

    parent = os.path.dirname(output_path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(stored, fh, ensure_ascii=False, indent=2)

    vector_path = vector_store.index_chunks(
        stored,
        chunks_path=output_path,
    )

    return IngestionResult(
        chunk_count=len(stored),
        files_processed=files_processed,
        output_path=output_path,
        vector_index_path=vector_path,
        pdf_files=pdf_files,
        text_files=text_files,
        skipped_files=skipped_files,
    )


# --- Backward-compatible API -----------------------------------------------


def retrieve(
    query: str,
    top_k: Optional[int] = None,
    chunks_path: Optional[str] = None,
) -> List[EvidenceChunk]:
    top_k = top_k or _config_mod.CONFIG.max_retrieved_docs
    candidates = retrieve_candidates(query, candidate_k=max(top_k * 3, 15), chunks_path=chunks_path)
    return rerank_candidates(query, candidates, top_k)


def retrieve_with_cache(
    query: str,
    scenario_mode: str,
    cache: Optional[Dict[str, List[EvidenceChunk]]] = None,
) -> Tuple[List[EvidenceChunk], bool]:
    if cache is None:
        cache = _RETRIEVAL_CACHE
    key = stable_hash(query, scenario_mode, "legacy")
    if key in cache:
        return cache[key], True
    chunks = retrieve(query)
    cache[key] = chunks
    return chunks, False


def clear_retrieval_cache() -> None:
    _RETRIEVAL_CACHE.clear()
    vector_store.reset_vector_store_cache()


def reload_rag_config() -> Dict[str, Any]:
    global ACTIVE_RAG_CONFIG
    ACTIVE_RAG_CONFIG = load_rag_config()
    return ACTIVE_RAG_CONFIG
