"""CLI to evaluate a named retrieval strategy against the labeled eval set.

Usage:
    python -m src.evaluation.run_retrieval_evals --retriever dense
    python -m src.evaluation.run_retrieval_evals --retriever bm25
    python -m src.evaluation.run_retrieval_evals --retriever hybrid
    python -m src.evaluation.run_retrieval_evals --retriever hybrid-rerank
"""

import argparse

from src.evaluation.retrieval_eval import evaluate_retrieval, load_eval_set, print_results
from src.retrieval.embedder import DEFAULT_MODEL, Embedder
from src.retrieval.retrievers import (
    bm25_retriever,
    dense_retriever,
    hybrid_retriever,
    reranked_retriever,
)
from src.retrieval.vector_store import VectorStore

BM25_PATH = "data/processed/bm25_index.pkl"


def build_retriever(args):
    needs_dense = args.retriever in ("dense", "hybrid", "hybrid-rerank")
    needs_bm25 = args.retriever in ("bm25", "hybrid", "hybrid-rerank")

    store = None
    dense_fn = bm25_fn = None
    if needs_dense:
        embedder = Embedder(model_name=args.model)
        store = VectorStore(collection_name=args.collection, embedding_dim=embedder.dim)
        dense_fn = dense_retriever(embedder, store)
    if needs_bm25:
        from src.retrieval.bm25_index import BM25Index

        bm25_fn = bm25_retriever(BM25Index.load(args.bm25_path))

    if args.retriever == "dense":
        return dense_fn
    if args.retriever == "bm25":
        return bm25_fn
    hybrid_fn = hybrid_retriever(dense_fn, bm25_fn)
    if args.retriever == "hybrid":
        return hybrid_fn

    from src.retrieval.reranker import Reranker

    return reranked_retriever(hybrid_fn, Reranker(), store.get_documents)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retriever",
        required=True,
        choices=["dense", "bm25", "hybrid", "hybrid-rerank"],
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--collection", default="complaints_bge")
    parser.add_argument("--bm25-path", default=BM25_PATH)
    args = parser.parse_args()

    retrieve_fn = build_retriever(args)
    results = evaluate_retrieval(load_eval_set(), retrieve_fn)

    label = args.retriever
    if args.retriever != "bm25":
        label += f" ({args.model}, collection={args.collection})"
    print_results(results, title=label)


if __name__ == "__main__":
    main()
