"""Deterministic (no-LLM) test of a cheap relief-vs-not signal.

Stage 3's LLM predictor beat no baseline (see eval-results.md). This asks a
narrower, LLM-free question: does the *outcome mix of the retrieved neighbors*
carry a usable signal for the binary question "will this complaint get any
relief (monetary or non-monetary)?"

Score for each query = fraction of its top-5 retrieved neighbors whose
company_response is a relief category. Label = the query's own outcome is a
relief category. We report ROC-AUC over the 80 eval-v2 queries — ROC-AUC
because it is threshold-free and reads the ranking quality of a score, which is
exactly what "is the neighbor relief-fraction informative" means. Everything
here is counting; no tokens are spent, so the Groq daily cap does not apply.

Small-sample caveat: 30 positives / 80 — a single query moves AUC by ~0.01,
so treat this as directional, not precise.
"""

import argparse
import json

from sklearn.metrics import roc_auc_score

from src.evaluation.retrieval_eval import load_eval_set
from src.retrieval.factory import DEFAULT_COLLECTION, make_retriever

RELIEF = {"Closed with monetary relief", "Closed with non-monetary relief"}
TOP_K = 5


def relief_fraction(outcomes: list[str]) -> float:
    if not outcomes:
        return 0.0
    return sum(o in RELIEF for o in outcomes) / len(outcomes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", default="data/eval/eval_set_v2.json")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    queries = load_eval_set(args.eval_set)
    stack = make_retriever("dense", collection=args.collection)

    labels, scores = [], []
    for q in queries:
        ids = stack.retrieve(q.query_narrative, q.query_complaint_id, args.top_k)
        records = stack.store.get_records(ids)
        outcomes = [records[i]["metadata"].get("company_response", "") for i in ids if i in records]
        scores.append(relief_fraction(outcomes))
        labels.append(1 if q.query_company_response in RELIEF else 0)

    auc = roc_auc_score(labels, scores)
    summary = {
        "n": len(labels),
        "n_relief_positive": sum(labels),
        "top_k": args.top_k,
        "metric": "roc_auc",
        "neighbor_relief_fraction_auc": round(auc, 4),
        "passes_0_65_gate": auc >= 0.65,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
