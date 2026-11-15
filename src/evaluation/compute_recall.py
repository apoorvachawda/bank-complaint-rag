import json
from pathlib import Path

from pydantic import BaseModel

from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import SearchResult, VectorStore

EVAL_SET_PATH = Path("data/eval/eval_set.json")


class EvalExample(BaseModel):
    eval_id: str
    query_complaint_id: str
    query_narrative: str
    query_product: str
    query_issue: str
    query_sub_issue: str
    query_company: str
    query_company_response: str
    similar_complaint_ids: list[str]
    similarity_reasoning: str
    num_similar_found: int


def load_eval_set(path: Path = EVAL_SET_PATH) -> dict[str, EvalExample]:
    """Load eval_set.json into a dict keyed by query_complaint_id for O(1) lookup."""
    with open(path) as f:
        raw_entries = json.load(f)

    eval_set: dict[str, EvalExample] = {}
    for raw_entry in raw_entries:
        example = EvalExample.model_validate(raw_entry)
        if example.query_complaint_id in eval_set:
            raise ValueError(
                f"Duplicate query_complaint_id in eval set: {example.query_complaint_id}"
            )
        eval_set[example.query_complaint_id] = example

    return eval_set


def load_vector_store(collection_name: str = "complaints", embedding_dim: int = 384) -> VectorStore:
    """Load the ChromaDB vector store."""
    return VectorStore(collection_name=collection_name, embedding_dim=embedding_dim)


def load_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Embedder:
    """Load the sentence transformer embedder."""
    return Embedder(model_name=model_name)


def recall_at_k(eval_item: EvalExample, results: list[SearchResult], k: int) -> float:
    """Recall@k with denominator min(k, |ground truth|).

    Ground-truth set sizes vary per query (1..n similar complaints). Dividing
    by the raw ground-truth size would cap recall@k below 1.0 whenever
    |ground truth| > k, making the aggregate incomparable across queries.
    With min(k, |GT|) a perfect top-k always scores 1.0.
    """
    ground_truth = set(eval_item.similar_complaint_ids)
    if not ground_truth:
        return 0.0
    retrieved_top_k = {r["id"] for r in results[:k]}
    matches = retrieved_top_k & ground_truth
    return len(matches) / min(k, len(ground_truth))


def _mean(values: list[float]) -> float:
    """Average a list of recall scores, or 0.0 if empty."""
    return sum(values) / len(values) if values else 0.0


def evaluate_retrieval(
    eval_examples: list[EvalExample],
    embedder: Embedder,
    vector_store: VectorStore,
    k_values: tuple[int, ...] = (5, 10),
) -> dict:
    """Run recall@k evaluation for a retriever over the full eval set.

    Retrieval is leave-one-out: each query's own complaint_id is stripped
    from its results before scoring, since it is not a valid "similar"
    match to itself. We over-fetch by one so that after self-removal the
    candidate list still covers the largest k.
    """
    max_k = max(k_values)

    embedded_queries = embedder.embed_batch([example.query_narrative for example in eval_examples])

    per_query_recalls: dict[int, list[float]] = {k: [] for k in k_values}
    by_product_recalls: dict[str, dict[int, list[float]]] = {}

    for i, eval_item in enumerate(eval_examples):
        results = vector_store.search(query_vector=embedded_queries[i], n=max_k + 1)
        results = [r for r in results if r["id"] != eval_item.query_complaint_id]
        results = results[:max_k]

        product_recalls = by_product_recalls.setdefault(
            eval_item.query_product, {k: [] for k in k_values}
        )

        for k in k_values:
            recall = recall_at_k(eval_item, results, k=k)
            per_query_recalls[k].append(recall)
            product_recalls[k].append(recall)

    overall = {f"recall@{k}": _mean(recalls) for k, recalls in per_query_recalls.items()}
    by_product = {
        product: {f"recall@{k}": _mean(recalls) for k, recalls in product_recalls.items()}
        for product, product_recalls in by_product_recalls.items()
    }

    return {
        "overall": overall,
        "by_product": by_product,
        "num_queries": len(eval_examples),
    }


if __name__ == "__main__":
    eval_set = load_eval_set()
    sample_set = list(eval_set.values())

    vector_store = load_vector_store()
    embedder = load_embedder()

    results = evaluate_retrieval(sample_set, embedder, vector_store)

    print(f"For entire eval set ({results['num_queries']} queries):")
    for metric, value in results["overall"].items():
        print(f"  {metric}: {value:.4f}")

    print("\nBy product (sorted by recall@5 descending):")
    products_sorted = sorted(
        results["by_product"].items(),
        key=lambda item: item[1]["recall@5"],
        reverse=True,
    )
    for product, metrics in products_sorted:
        metrics_str = ", ".join(f"{name}: {value:.4f}" for name, value in metrics.items())
        print(f"  {product}: {metrics_str}")
