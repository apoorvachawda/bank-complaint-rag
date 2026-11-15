"""Rebuild eval-set ground truth with POOLED candidates (v2).

Why v2 exists: v1 mined candidate pairs exclusively from MiniLM dense
top-20 neighbors, so the labels could never contain a complaint MiniLM
would not retrieve. That made cross-retriever comparison circular —
MiniLM dense scored 0.58 recall@5 by construction while BM25 (0.07) and
BGE (0.29) were penalized for surfacing different-but-possibly-similar
complaints. Classic pooling bias; the standard fix (TREC-style) is to
pool candidates from ALL systems under comparison and judge the union.

What stays identical to v1 (deliberately):
- The same 80 query complaints (loaded from eval_set.json).
- The same LLM judge system prompt and per-candidate format.
- The same output schema.

What changes:
- Candidates = union of MiniLM-dense, BGE-dense, and BM25 top-N per query.
- The judge no longer sees a cosine-similarity number per candidate
  (v1 showed it, anchoring the judge toward dense-favored candidates).
- Judge model is openai/gpt-oss-120b — v1 used llama-3.3-70b, since
  retired from Groq.
"""

import argparse
import json
import logging
import time
from pathlib import Path

from dotenv import load_dotenv

from src.evaluation.retrieval_eval import load_eval_set
from src.pipeline.llm import chat_json
from src.retrieval.factory import make_retriever

load_dotenv()
logger = logging.getLogger(__name__)

POOL_PER_RETRIEVER = 15

# Identical to v1 (build_eval_set.SYSTEM_PROMPT)
SYSTEM_PROMPT = (
    "You are evaluating whether complaint pairs are SIMILAR enough that retrieving "
    "one would be useful context when handling the other.\n\n"
    "Similar means: same type of banking problem. A bank teller seeing the retrieved "
    "complaint would gain useful context about how this type of issue is typically handled.\n\n"
    "NOT similar: same product category but fundamentally different problems "
    "(e.g., overdraft fee vs account closure).\n\n"
    "You will receive a query complaint and a list of candidates. "
    "For each candidate, judge if it's similar.\n\n"
    "Respond with JSON:\n"
    '{"similar_ids": ["id1", "id2", ...], '
    '"reasoning": "brief explanation of what makes them similar"}'
)


def build_prompt(query, candidates: list[dict]) -> str:
    lines = [
        f"QUERY COMPLAINT (ID: {query.query_complaint_id}):",
        f"Product: {query.query_product}",
        f"Issue: {query.query_issue}",
        f"Narrative: {query.query_narrative[:1000]}",
        "",
        "CANDIDATES:",
    ]
    for c in candidates:
        lines.append(f"[{c['id']}] Issue: {c['issue']} | Narrative: {c['narrative'][:300]}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1", type=Path, default=Path("data/eval/eval_set.json"))
    parser.add_argument("--output", type=Path, default=Path("data/eval/eval_set_v2.json"))
    parser.add_argument("--pool", type=int, default=POOL_PER_RETRIEVER)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    queries = load_eval_set(args.v1)
    query_ids = {q.query_complaint_id for q in queries}

    logger.info("Building the three candidate retrievers...")
    minilm = make_retriever(
        "dense", model="sentence-transformers/all-MiniLM-L6-v2", collection="complaints"
    )
    bge = make_retriever("dense", model="BAAI/bge-small-en-v1.5", collection="complaints_bge")
    bm25 = make_retriever("bm25")

    entries = []
    for i, q in enumerate(queries, 1):
        pooled: list[str] = []
        for stack in (minilm, bge, bm25):
            for cid in stack.retrieve(q.query_narrative, q.query_complaint_id, args.pool):
                # exclude other eval queries (v1 rule: no cross-contamination)
                if cid not in pooled and cid not in query_ids:
                    pooled.append(cid)

        records = bge.store.get_records(pooled)
        candidates = [
            {
                "id": cid,
                "narrative": records[cid]["document"],
                "issue": records[cid]["metadata"].get("issue", ""),
            }
            for cid in pooled
            if cid in records
        ]

        raw = chat_json(
            system=SYSTEM_PROMPT,
            user=build_prompt(q, candidates),
            name="evalset-v2-judge",
        )
        candidate_ids = {c["id"] for c in candidates}
        similar_ids = [str(s) for s in raw.get("similar_ids", []) if str(s) in candidate_ids]

        entries.append(
            {
                "eval_id": q.eval_id,
                "query_complaint_id": q.query_complaint_id,
                "query_narrative": q.query_narrative,
                "query_product": q.query_product,
                "query_issue": q.query_issue,
                "query_sub_issue": q.query_sub_issue,
                "query_company": q.query_company,
                "query_company_response": q.query_company_response,
                "similar_complaint_ids": similar_ids,
                "similarity_reasoning": raw.get("reasoning", ""),
                "num_similar_found": len(similar_ids),
                "candidate_pool_size": len(candidates),
            }
        )

        if i % 10 == 0:
            logger.info("Judged %d/%d queries (checkpoint save)", i, len(queries))
            args.output.write_text(json.dumps(entries, indent=2, ensure_ascii=False))
        time.sleep(0.5)

    args.output.write_text(json.dumps(entries, indent=2, ensure_ascii=False))

    sizes = [e["num_similar_found"] for e in entries]
    nonzero = [s for s in sizes if s > 0]
    logger.info(
        "Done. %d entries, %d with similar found; per-query min/mean/max = %d/%.1f/%d",
        len(entries),
        len(nonzero),
        min(nonzero) if nonzero else 0,
        sum(nonzero) / len(nonzero) if nonzero else 0,
        max(nonzero) if nonzero else 0,
    )


if __name__ == "__main__":
    main()
