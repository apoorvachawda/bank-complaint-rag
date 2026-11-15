"""Retriever-agnostic evaluation: recall@k, MRR, per-product breakdown.

Any retrieval strategy (dense, BM25, hybrid, reranked) is evaluated through
the same RetrieveFn interface so results are directly comparable. With only
80 labeled queries, differences of a few points are within noise — treat
small deltas as ties, not wins.
"""

import json
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import BaseModel

EVAL_SET_PATH = Path("data/eval/eval_set.json")

# (query_narrative, exclude_id, n) -> ranked complaint ids, best first.
# Implementations must drop exclude_id (leave-one-out: the query's own
# complaint is not a valid "similar" match to itself).
RetrieveFn = Callable[[str, str, int], list[str]]


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


def load_eval_set(path: Path = EVAL_SET_PATH) -> list[EvalExample]:
    with open(path) as f:
        raw_entries = json.load(f)

    examples = [EvalExample.model_validate(e) for e in raw_entries]
    seen_ids = {e.query_complaint_id for e in examples}
    if len(seen_ids) != len(examples):
        raise ValueError("Duplicate query_complaint_id in eval set")
    return examples


def recall_at_k(ground_truth: set[str], retrieved_ids: list[str], k: int) -> float:
    """Recall@k with denominator min(k, |ground truth|).

    Ground-truth set sizes vary per query; with a raw |GT| denominator a
    perfect top-k could never reach 1.0 when |GT| > k, making scores
    incomparable across queries (see docs/eval-results.md).
    """
    if not ground_truth:
        return 0.0
    matches = set(retrieved_ids[:k]) & ground_truth
    return len(matches) / min(k, len(ground_truth))


def reciprocal_rank(ground_truth: set[str], retrieved_ids: list[str]) -> float:
    """1/rank of the first relevant result, 0.0 if none retrieved."""
    for rank, id_ in enumerate(retrieved_ids, start=1):
        if id_ in ground_truth:
            return 1.0 / rank
    return 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_retrieval(
    eval_examples: list[EvalExample],
    retrieve_fn: RetrieveFn,
    k_values: Sequence[int] = (5, 10),
) -> dict:
    max_k = max(k_values)
    mrr_key = f"mrr@{max_k}"

    per_query: dict[str, list[float]] = {f"recall@{k}": [] for k in k_values}
    per_query[mrr_key] = []
    by_product: dict[str, dict[str, list[float]]] = {}

    for ex in eval_examples:
        retrieved = retrieve_fn(ex.query_narrative, ex.query_complaint_id, max_k)
        ground_truth = set(ex.similar_complaint_ids)

        metrics = {f"recall@{k}": recall_at_k(ground_truth, retrieved, k) for k in k_values}
        metrics[mrr_key] = reciprocal_rank(ground_truth, retrieved[:max_k])

        product_metrics = by_product.setdefault(ex.query_product, {name: [] for name in per_query})
        for name, value in metrics.items():
            per_query[name].append(value)
            product_metrics[name].append(value)

    return {
        "overall": {name: _mean(vals) for name, vals in per_query.items()},
        "by_product": {
            product: {name: _mean(vals) for name, vals in metrics.items()}
            for product, metrics in by_product.items()
        },
        "num_queries": len(eval_examples),
    }


def print_results(results: dict, title: str) -> None:
    print(f"\n{title} ({results['num_queries']} queries)")
    for metric, value in results["overall"].items():
        print(f"  {metric}: {value:.4f}")
    print("\nBy product (sorted by recall@5 descending):")
    for product, metrics in sorted(
        results["by_product"].items(), key=lambda kv: kv[1]["recall@5"], reverse=True
    ):
        metrics_str = ", ".join(f"{name}: {value:.4f}" for name, value in metrics.items())
        print(f"  {product}: {metrics_str}")
