"""Retrieval strategies behind a single interface.

Every factory returns a RetrieveFn: (query_narrative, exclude_id, n) ->
ranked complaint ids. The eval harness and the pipeline consume retrievers
through this interface only, so strategies are swappable and comparable.
"""

from collections.abc import Callable

from src.retrieval.bm25_index import BM25Index
from src.retrieval.embedder import Embedder
from src.retrieval.reranker import Reranker
from src.retrieval.vector_store import VectorStore

RetrieveFn = Callable[[str, str, int], list[str]]

# How many candidates each leg contributes before fusion/reranking. Recall@k
# upstream bounds everything downstream — a reranker cannot recover a
# complaint the candidate pool never contained.
CANDIDATE_POOL = 50

# Standard RRF constant from Cormack et al.; dampens the advantage of
# rank-1 vs rank-5 so neither leg dominates the fusion.
RRF_K = 60


def dense_retriever(embedder: Embedder, store: VectorStore) -> RetrieveFn:
    def retrieve(narrative: str, exclude_id: str, n: int) -> list[str]:
        results = store.search(embedder.embed_query(narrative), n=n + 1)
        ids = [r["id"] for r in results if r["id"] != exclude_id]
        return ids[:n]

    return retrieve


def bm25_retriever(index: BM25Index) -> RetrieveFn:
    def retrieve(narrative: str, exclude_id: str, n: int) -> list[str]:
        ids = [id_ for id_, _ in index.search(narrative, n + 1) if id_ != exclude_id]
        return ids[:n]

    return retrieve


def hybrid_retriever(
    dense_fn: RetrieveFn,
    bm25_fn: RetrieveFn,
    pool: int = CANDIDATE_POOL,
    rrf_k: int = RRF_K,
) -> RetrieveFn:
    """Reciprocal Rank Fusion of dense + BM25 rankings.

    RRF fuses on ranks, not scores — cosine similarities and BM25 scores
    live on incomparable scales, and score normalization is brittle
    (sensitive to outliers and per-query score ranges). See ADR-010.
    """

    def retrieve(narrative: str, exclude_id: str, n: int) -> list[str]:
        fused: dict[str, float] = {}
        rankings = (dense_fn(narrative, exclude_id, pool), bm25_fn(narrative, exclude_id, pool))
        for ranking in rankings:
            for rank, id_ in enumerate(ranking):
                fused[id_] = fused.get(id_, 0.0) + 1.0 / (rrf_k + rank + 1)
        ranked = sorted(fused, key=lambda id_: fused[id_], reverse=True)
        return ranked[:n]

    return retrieve


def reranked_retriever(
    base_fn: RetrieveFn,
    reranker: Reranker,
    get_texts: Callable[[list[str]], dict[str, str]],
    pool: int = CANDIDATE_POOL,
) -> RetrieveFn:
    """Cross-encoder reranking of a base retriever's candidate pool.

    The base retriever casts a wide net (pool candidates); the cross-encoder
    reads query and candidate together — far more accurate than comparing
    two independently-computed vectors — and reorders the pool.
    """

    def retrieve(narrative: str, exclude_id: str, n: int) -> list[str]:
        candidates = base_fn(narrative, exclude_id, pool)
        texts = get_texts(candidates)
        candidates = [c for c in candidates if c in texts]
        scores = reranker.score(narrative, [texts[c] for c in candidates])
        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        return [candidates[i] for i in order[:n]]

    return retrieve
