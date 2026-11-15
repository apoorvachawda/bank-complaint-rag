"""Stage 3 resolution-prediction eval on a held-out test sample.

Design constraints (non-negotiable for a defensible number):
- The complaint being predicted is NEVER retrievable as its own evidence:
  its id is excluded from retrieval results (exclude_complaint_id).
- Resolutions are heavily imbalanced ("Closed with explanation" dominates),
  so accuracy alone is misleading — we report macro-F1 and compare BOTH
  metrics against a majority-class baseline. Beating the baseline on
  accuracy but not macro-F1 means the model learned nothing beyond the
  class prior.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.pipeline.graph import build_pipeline
from src.pipeline.models import RESOLUTION_CATEGORIES, PipelineState
from src.retrieval.factory import (
    DEFAULT_BM25_PATH,
    DEFAULT_COLLECTION,
    RETRIEVER_CHOICES,
    make_retriever,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

CORPUS_PATH = "data/processed/complaints_deduped.jsonl"
RESULTS_PATH = "data/eval/prediction_results.json"


def sample_test_split(corpus_path: str, n: int, seed: int) -> pd.DataFrame:
    df = pd.read_json(corpus_path, lines=True)
    df = df.dropna(subset=["narrative", "company_response"])
    df = df[df["company_response"].isin(RESOLUTION_CATEGORIES)]
    return df.sample(n=n, random_state=seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retriever", default="hybrid-rerank", choices=RETRIEVER_CHOICES)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--bm25-path", default=DEFAULT_BM25_PATH)
    parser.add_argument(
        "--sleep", type=float, default=1.0, help="Pause between queries (rate limits)"
    )
    parser.add_argument("--output", default=RESULTS_PATH)
    args = parser.parse_args()

    test_df = sample_test_split(CORPUS_PATH, n=args.n, seed=args.seed)
    logger.info(
        "Test split: %d complaints, class distribution:\n%s",
        len(test_df),
        test_df["company_response"].value_counts().to_string(),
    )

    # Majority class measured on the corpus EXCLUDING the test split —
    # the baseline must not peek at test labels either.
    corpus = pd.read_json(CORPUS_PATH, lines=True)
    corpus = corpus[corpus["company_response"].isin(RESOLUTION_CATEGORIES)]
    train = corpus[~corpus["complaint_id"].isin(test_df["complaint_id"])]
    majority_class = train["company_response"].mode()[0]
    logger.info("Majority class (train): %s", majority_class)

    stack = make_retriever(args.retriever, collection=args.collection, bm25_path=args.bm25_path)
    pipeline = build_pipeline(stack.retrieve, stack.store.get_records)

    records = []
    for i, row in enumerate(test_df.itertuples(), start=1):
        complaint_id = str(row.complaint_id)
        try:
            final = pipeline.invoke(
                PipelineState(
                    complaint_text=str(row.narrative),
                    exclude_complaint_id=complaint_id,
                )
            )
            prediction = final["prediction"]
            records.append(
                {
                    "complaint_id": complaint_id,
                    "actual": row.company_response,
                    "predicted": prediction.predicted_resolution,
                    "confidence": prediction.confidence,
                    "cited_evidence_ids": prediction.cited_evidence_ids,
                    "evidence_outcomes": [e.company_response for e in final["evidence"]],
                }
            )
        except Exception:
            logger.exception("Pipeline failed on %s — skipping", complaint_id)
        if i % 10 == 0:
            logger.info("Processed %d/%d", i, len(test_df))
        time.sleep(args.sleep)

    y_true = [r["actual"] for r in records]
    y_pred = [r["predicted"] for r in records]
    y_majority = [majority_class] * len(records)

    summary = {
        "n_requested": args.n,
        "n_completed": len(records),
        "seed": args.seed,
        "retriever": args.retriever,
        "majority_class": majority_class,
        "model": {
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        },
        "majority_baseline": {
            "accuracy": accuracy_score(y_true, y_majority),
            "macro_f1": f1_score(y_true, y_majority, average="macro", zero_division=0),
        },
    }

    print(json.dumps(summary, indent=2))
    print("\nPer-class report (model):")
    print(classification_report(y_true, y_pred, zero_division=0))

    Path(args.output).write_text(json.dumps({"summary": summary, "records": records}, indent=2))
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
