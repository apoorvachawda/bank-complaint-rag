from src.retrieval.retrievers import hybrid_retriever, reranked_retriever


def test_rrf_fusion_prefers_items_ranked_by_both_legs():
    dense = lambda q, ex, n: ["a", "b", "c", "d"]  # noqa: E731
    bm25 = lambda q, ex, n: ["c", "e", "a", "f"]  # noqa: E731
    fused = hybrid_retriever(dense, bm25, pool=4)("query", "", 4)
    # "a" (ranks 1+3) and "c" (ranks 3+1) appear in both legs -> top two
    assert set(fused[:2]) == {"a", "c"}


def test_rrf_single_leg_items_still_ranked():
    dense = lambda q, ex, n: ["a"]  # noqa: E731
    bm25 = lambda q, ex, n: []  # noqa: E731
    assert hybrid_retriever(dense, bm25)("query", "", 5) == ["a"]


class FakeReranker:
    def score(self, query, documents):
        # reverse the incoming order
        return list(range(len(documents)))


def test_reranker_reorders_candidates():
    base = lambda q, ex, n: ["a", "b", "c"]  # noqa: E731
    texts = lambda ids: {i: f"text-{i}" for i in ids}  # noqa: E731
    result = reranked_retriever(base, FakeReranker(), texts, pool=3)("query", "", 3)
    assert result == ["c", "b", "a"]


def test_reranker_drops_candidates_with_missing_text():
    base = lambda q, ex, n: ["a", "b", "c"]  # noqa: E731
    texts = lambda ids: {"a": "ta", "c": "tc"}  # noqa: E731
    result = reranked_retriever(base, FakeReranker(), texts, pool=3)("query", "", 3)
    assert set(result) == {"a", "c"}
