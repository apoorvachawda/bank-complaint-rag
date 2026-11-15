"""Build a named retrieval strategy with its backing stores.

Single construction point shared by the eval CLIs and the API service, so
"which retriever is live" is always a one-string config change, and the
vector backend (local Chroma vs Qdrant Cloud) is an env-var swap
(VECTOR_BACKEND=chroma|qdrant, see ADR-002).
"""

import os
from dataclasses import dataclass

from src.retrieval.embedder import DEFAULT_MODEL, Embedder
from src.retrieval.retrievers import (
    RetrieveFn,
    bm25_retriever,
    dense_retriever,
    hybrid_retriever,
    reranked_retriever,
)

DEFAULT_COLLECTION = "complaints_bge"
DEFAULT_BM25_PATH = "data/processed/bm25_index.pkl"

RETRIEVER_CHOICES = ["dense", "bm25", "hybrid", "hybrid-rerank"]


def make_store(collection: str, embedding_dim: int, backend: str | None = None):
    backend = backend or os.environ.get("VECTOR_BACKEND", "chroma")
    if backend == "qdrant":
        from src.retrieval.qdrant_store import QdrantStore

        return QdrantStore(collection_name=collection, embedding_dim=embedding_dim)
    from src.retrieval.vector_store import VectorStore

    return VectorStore(collection_name=collection, embedding_dim=embedding_dim)


@dataclass
class RetrievalStack:
    retrieve: RetrieveFn
    store: object | None  # VectorStore | QdrantStore; None for bm25-only


def make_retriever(
    name: str,
    model: str = DEFAULT_MODEL,
    collection: str = DEFAULT_COLLECTION,
    bm25_path: str = DEFAULT_BM25_PATH,
    backend: str | None = None,
) -> RetrievalStack:
    needs_dense = name in ("dense", "hybrid", "hybrid-rerank")
    needs_bm25 = name in ("bm25", "hybrid", "hybrid-rerank")

    store = None
    dense_fn = bm25_fn = None
    if needs_dense:
        embedder = Embedder(model_name=model)
        store = make_store(collection, embedder.dim, backend)
        dense_fn = dense_retriever(embedder, store)
    if needs_bm25:
        from src.retrieval.bm25_index import BM25Index

        bm25_fn = bm25_retriever(BM25Index.load(bm25_path))

    if name == "dense":
        return RetrievalStack(dense_fn, store)
    if name == "bm25":
        return RetrievalStack(bm25_fn, None)

    hybrid_fn = hybrid_retriever(dense_fn, bm25_fn)
    if name == "hybrid":
        return RetrievalStack(hybrid_fn, store)
    if name == "hybrid-rerank":
        from src.retrieval.reranker import Reranker

        return RetrievalStack(reranked_retriever(hybrid_fn, Reranker(), store.get_documents), store)
    raise ValueError(f"Unknown retriever: {name!r} (choices: {RETRIEVER_CHOICES})")
