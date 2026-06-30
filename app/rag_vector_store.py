"""Persistent Chroma vector index for knowledge-base retrieval."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from . import config as _config_mod

_COLLECTION = "knowledge_base"


def reset_vector_store_cache() -> None:
    """No-op placeholder; kept for test isolation hooks."""
    return None


def _resolve_chroma_path(
    chunks_path: Optional[str] = None,
    chroma_path: Optional[str] = None,
) -> str:
    if chroma_path:
        return chroma_path
    if chunks_path:
        return os.path.join(os.path.dirname(os.path.abspath(chunks_path)), "chroma")
    return _config_mod.CONFIG.rag_chroma_path


def _embedding_function():
    try:
        from chromadb.utils import embedding_functions
    except ImportError as e:
        raise RuntimeError(
            "Chroma vector RAG requires the 'chromadb' package. Run: pip install chromadb"
        ) from e

    cfg = _config_mod.CONFIG
    if cfg.openai_api_key:
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=cfg.openai_api_key,
            model_name=cfg.rag_embedding_model,
        )
    return embedding_functions.DefaultEmbeddingFunction()


def _client(persist_path: str):
    import chromadb

    parent = os.path.dirname(persist_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return chromadb.PersistentClient(path=persist_path)


def _get_collection(client, *, create: bool = True):
    ef = _embedding_function()
    if create:
        return client.get_or_create_collection(
            name=_COLLECTION,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
    return client.get_collection(name=_COLLECTION, embedding_function=ef)


def index_chunks(
    chunks: List[Dict[str, Any]],
    *,
    chunks_path: Optional[str] = None,
    chroma_path: Optional[str] = None,
) -> str:
    """Rebuild the Chroma collection from normalized chunk dicts."""
    path = _resolve_chroma_path(chunks_path, chroma_path)
    client = _client(path)

    try:
        client.delete_collection(_COLLECTION)
    except Exception:
        pass

    if not chunks:
        return path

    collection = _get_collection(client, create=True)
    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for ch in chunks:
        cid = str(ch.get("chunk_id") or "")
        text = str(ch.get("text") or "")
        if not cid or not text.strip():
            continue
        ids.append(cid)
        documents.append(text)
        metadatas.append(
            {
                "source_path": str(ch.get("source_path") or "")[:500],
                "source_name": str(ch.get("source_name") or "")[:200],
                "domain": str(ch.get("domain") or "general"),
                "source_type": str(ch.get("source_type") or "unknown"),
                "period": str(ch.get("period") or "unknown"),
                "char_count": int(ch.get("char_count") or len(text)),
            }
        )

    batch_size = 64
    for start in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[start : start + batch_size],
            documents=documents[start : start + batch_size],
            metadatas=metadatas[start : start + batch_size],
        )
    return path


def collection_count(
    *,
    chunks_path: Optional[str] = None,
    chroma_path: Optional[str] = None,
) -> int:
    path = _resolve_chroma_path(chunks_path, chroma_path)
    if not os.path.isdir(path):
        return 0
    try:
        client = _client(path)
        collection = _get_collection(client, create=False)
        return int(collection.count())
    except Exception:
        return 0


def query_chunks(
    query: str,
    *,
    n_results: int,
    where: Optional[Dict[str, Any]] = None,
    chunks_path: Optional[str] = None,
    chroma_path: Optional[str] = None,
) -> List[Tuple[Dict[str, Any], float]]:
    """Return (chunk_dict, similarity_score) pairs from the persistent index."""
    path = _resolve_chroma_path(chunks_path, chroma_path)
    if not os.path.isdir(path):
        return []

    try:
        client = _client(path)
        collection = _get_collection(client, create=False)
    except Exception:
        return []

    total = collection.count()
    if total == 0:
        return []

    limit = max(1, min(n_results, total))
    kwargs: Dict[str, Any] = {
        "query_texts": [query],
        "n_results": limit,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    try:
        raw = collection.query(**kwargs)
    except Exception:
        if where:
            kwargs.pop("where", None)
            try:
                raw = collection.query(**kwargs)
            except Exception:
                return []
        else:
            return []

    ids = (raw.get("ids") or [[]])[0]
    docs = (raw.get("documents") or [[]])[0]
    metas = (raw.get("metadatas") or [[]])[0]
    dists = (raw.get("distances") or [[]])[0]

    out: List[Tuple[Dict[str, Any], float]] = []
    for cid, doc, meta, dist in zip(ids, docs, metas, dists):
        meta = meta or {}
        try:
            score = max(0.0, 1.0 - float(dist))
        except (TypeError, ValueError):
            score = 0.0
        chunk = {
            "chunk_id": cid,
            "text": doc or "",
            "source_path": meta.get("source_path") or "",
            "source_name": meta.get("source_name") or "",
            "domain": meta.get("domain") or "general",
            "source_type": meta.get("source_type") or "unknown",
            "period": meta.get("period") or "unknown",
            "char_count": int(meta.get("char_count") or len(doc or "")),
        }
        out.append((chunk, score))
    return out
