"""CLI to evaluate a named retrieval strategy against the labeled eval set.

Usage:
    python -m src.evaluation.run_retrieval_evals --retriever dense
    python -m src.evaluation.run_retrieval_evals --retriever bm25
    python -m src.evaluation.run_retrieval_evals --retriever hybrid
    python -m src.evaluation.run_retrieval_evals --retriever hybrid-rerank
"""

import argparse

from src.evaluation.retrieval_eval import evaluate_retrieval, load_eval_set, print_results
from src.retrieval.embedder import DEFAULT_MODEL
from src.retrieval.factory import (
    DEFAULT_BM25_PATH,
    DEFAULT_COLLECTION,
    RETRIEVER_CHOICES,
    make_retriever,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", required=True, choices=RETRIEVER_CHOICES)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--bm25-path", default=DEFAULT_BM25_PATH)
    args = parser.parse_args()

    stack = make_retriever(
        args.retriever, model=args.model, collection=args.collection, bm25_path=args.bm25_path
    )
    results = evaluate_retrieval(load_eval_set(), stack.retrieve)

    label = args.retriever
    if args.retriever != "bm25":
        label += f" ({args.model}, collection={args.collection})"
    print_results(results, title=label)


if __name__ == "__main__":
    main()
