"""
Build the eval dataset for testing retrieval quality and resolution prediction accuracy.

Two things are being evaluated:
  1. RETRIEVAL — Recall@k: given a query complaint, does the retriever surface
     genuinely similar complaints from the corpus?
  2. RESOLUTION PREDICTION — accuracy: does Stage 3 correctly predict the
     company_response category based on retrieved similar complaints?
     Ground truth = actual company_response field from CFPB data.

Pipeline
────────
  Phase 1: Sample 80 query complaints (16 per product, balanced by issue + response)
  Phase 2: Embed all complaints per product, find top-20 cosine-similar candidates
           per query. Cache embeddings to data/eval/embeddings_cache.npz.
  Phase 3: LLM (Groq llama-3.3-70b) verifies which candidates are genuinely similar.
  Phase 4: Write data/eval/eval_set.json and print summary.

Usage
─────
  python -m src.data.build_eval_set
  python -m src.data.build_eval_set --skip-embedding   # use cached embeddings
  python -m src.data.build_eval_set --skip-llm         # skip LLM, use sim >= 0.6
  python -m src.data.build_eval_set --num-queries 20   # quick test run
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import signal
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MIN_NARRATIVE_LEN = 200
CANDIDATE_POOL_SIZE = 20  # top-N by cosine sim passed to LLM
MIN_SIM_FOR_CANDIDATES = 0.4  # below this, too dissimilar to be useful
FALLBACK_SIM_THRESHOLD = 0.6  # used when --skip-llm or LLM parse fails
MAX_ISSUES_PER_QUERY = 3  # no more than 3 queries from the same issue

PRODUCTS = [
    "Credit card or prepaid card",
    "Checking or savings account",
    "Money transfer, virtual currency, or money service",
    "Mortgage",
    "Vehicle loan or lease",
]

# Target response distribution per product (approximate — take what's available)
RESPONSE_TARGETS = {
    "Closed with explanation": 10,
    "Closed with monetary relief": 3,
    "Closed with non-monetary relief": 3,
}

# ── I/O ───────────────────────────────────────────────────────────────────────


def load_complaints(path: Path) -> list[dict]:
    complaints = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                complaints.append(json.loads(line))
    return complaints


def save_eval_set(output_path: Path, entries: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2, ensure_ascii=False)
    logger.info("Saved %d entries to %s", len(entries), output_path)


# ── Phase 1: Sample query complaints ─────────────────────────────────────────


def sample_queries(
    complaints: list[dict],
    queries_per_product: int,
) -> list[dict]:
    """
    Sample queries_per_product complaints per product, balancing across:
    - issue types (max MAX_ISSUES_PER_QUERY per issue)
    - company_response values (following RESPONSE_TARGETS)
    - narrative length (>= MIN_NARRATIVE_LEN)
    """
    by_product: dict[str, list[dict]] = defaultdict(list)
    for c in complaints:
        product = c.get("product") or ""
        narrative = (c.get("narrative") or "").strip()
        if product in PRODUCTS and len(narrative) >= MIN_NARRATIVE_LEN:
            by_product[product].append(c)

    selected: list[dict] = []

    for product in PRODUCTS:
        pool = by_product[product]
        random.shuffle(pool)

        issue_counts: Counter[str] = Counter()
        response_counts: Counter[str] = Counter()
        product_selected: list[dict] = []

        for c in pool:
            if len(product_selected) >= queries_per_product:
                break

            issue = c.get("issue") or ""
            response = c.get("company_response") or ""

            if issue_counts[issue] >= MAX_ISSUES_PER_QUERY:
                continue

            # Soft cap on response types — still include if we have room
            target = RESPONSE_TARGETS.get(response, 0)
            if target > 0 and response_counts[response] >= target:
                # Only skip if we're not desperate for entries
                remaining_slots = queries_per_product - len(product_selected)
                remaining_pool = len(pool) - pool.index(c) - 1
                if remaining_pool > remaining_slots * 2:
                    continue

            product_selected.append(c)
            issue_counts[issue] += 1
            response_counts[response] += 1

        selected.extend(product_selected)
        logger.info(
            "Sampled %d queries for product '%s' (pool size: %d)",
            len(product_selected),
            product,
            len(pool),
        )

    return selected


# ── Phase 2: Embed + find candidates ─────────────────────────────────────────


def load_or_build_embeddings(
    complaints: list[dict],
    cache_path: Path,
    skip_embedding: bool,
) -> tuple[np.ndarray, list[str]]:
    """
    Returns (embeddings, complaint_ids) arrays aligned by index.
    Loads from cache if available and skip_embedding=True, otherwise re-embeds.
    """
    if skip_embedding and cache_path.exists():
        logger.info("Loading cached embeddings from %s", cache_path)
        data = np.load(cache_path, allow_pickle=True)
        embeddings = data["embeddings"]
        complaint_ids = data["complaint_ids"].tolist()
        logger.info("Loaded %d embeddings from cache", len(complaint_ids))
        return embeddings, complaint_ids

    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)

    narratives = [c.get("narrative") or "" for c in complaints]
    complaint_ids = [str(c["complaint_id"]) for c in complaints]

    logger.info("Embedding %d complaints (this may take a few minutes)...", len(narratives))
    embeddings = model.encode(
        narratives,
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        embeddings=embeddings,
        complaint_ids=np.array(complaint_ids),
    )
    logger.info("Saved embeddings cache to %s", cache_path)
    return embeddings, complaint_ids


def find_candidates(
    query_complaint: dict,
    all_complaints: list[dict],
    all_embeddings: np.ndarray,
    all_ids: list[str],
    id_to_index: dict[str, int],
) -> list[dict]:
    """
    Find top candidates for a query within the same product category.
    Returns list of candidate dicts with 'complaint_id', 'similarity', 'narrative', 'issue'.
    """
    query_id = str(query_complaint["complaint_id"])
    product = query_complaint.get("product") or ""

    # Build index of same-product complaints (excluding the query itself)
    same_product_indices = [
        i
        for i, c in enumerate(all_complaints)
        if (c.get("product") or "") == product and str(c["complaint_id"]) != query_id
    ]

    if not same_product_indices:
        return []

    query_idx = id_to_index.get(query_id)
    if query_idx is None:
        return []

    query_emb = all_embeddings[query_idx : query_idx + 1]
    candidate_embs = all_embeddings[same_product_indices]

    sims = cosine_similarity(query_emb, candidate_embs)[0]

    # Rank by similarity, apply minimum threshold
    ranked = sorted(
        zip(sims, same_product_indices),
        key=lambda x: x[0],
        reverse=True,
    )
    top = [(sim, idx) for sim, idx in ranked if sim >= MIN_SIM_FOR_CANDIDATES]
    top = top[:CANDIDATE_POOL_SIZE]

    return [
        {
            "complaint_id": str(all_complaints[idx]["complaint_id"]),
            "similarity": float(sim),
            "narrative": all_complaints[idx].get("narrative") or "",
            "issue": all_complaints[idx].get("issue") or "",
        }
        for sim, idx in top
    ]


# ── Phase 3: LLM verification ─────────────────────────────────────────────────


def _build_llm_prompt(query: dict, candidates: list[dict]) -> str:
    lines = [
        f"QUERY COMPLAINT (ID: {query['complaint_id']}):",
        f"Product: {query.get('product') or ''}",
        f"Issue: {query.get('issue') or ''}",
        f"Narrative: {(query.get('narrative') or '')[:1000]}",
        "",
        "CANDIDATES:",
    ]
    for c in candidates:
        lines.append(
            f"[{c['complaint_id']}] Issue: {c['issue']} | "
            f"Similarity: {c['similarity']:.2f} | "
            f"Narrative: {c['narrative'][:300]}"
        )
    return "\n".join(lines)


SYSTEM_PROMPT = (
    "You are evaluating whether complaint pairs are SIMILAR enough that retrieving "
    "one would be useful context when handling the other.\n\n"
    "Similar means: same type of banking problem. A bank teller seeing the retrieved "
    "complaint would gain useful context about how this type of issue is typically handled.\n\n"
    "NOT similar: same product category but fundamentally different problems "
    "(e.g., overdraft fee vs account closure).\n\n"
    "You will receive a query complaint and up to 10 candidates. "
    "For each candidate, judge if it's similar.\n\n"
    "Respond with JSON:\n"
    '{"similar_ids": ["id1", "id2", ...], '
    '"reasoning": "brief explanation of what makes them similar"}'
)


def verify_with_llm(
    client,  # groq.Groq
    query: dict,
    candidates: list[dict],
) -> tuple[list[str], str]:
    """
    Returns (similar_ids, reasoning). Falls back to empty list on error.
    """
    import groq as _groq

    user_content = _build_llm_prompt(query, candidates)
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        result = json.loads(response.choices[0].message.content)
        similar_ids = [str(i) for i in result.get("similar_ids", [])]
        reasoning = result.get("reasoning", "")
        return similar_ids, reasoning
    except (_groq.APIError, json.JSONDecodeError, KeyError) as exc:
        logger.warning("LLM call failed for query %s: %s", query["complaint_id"], exc)
        return [], ""


# ── Phase 4: Assemble eval entries ────────────────────────────────────────────


def make_eval_entry(
    eval_num: int,
    query: dict,
    similar_ids: list[str],
    reasoning: str,
) -> dict:
    return {
        "eval_id": f"eval_{eval_num:03d}",
        "query_complaint_id": str(query["complaint_id"]),
        "query_narrative": (query.get("narrative") or "").strip(),
        "query_product": query.get("product") or "",
        "query_issue": query.get("issue") or "",
        "query_sub_issue": query.get("sub_issue") or "",
        "query_company": query.get("company") or "",
        "query_company_response": query.get("company_response") or "",
        "similar_complaint_ids": similar_ids,
        "similarity_reasoning": reasoning,
        "num_similar_found": len(similar_ids),
    }


# ── Summary ───────────────────────────────────────────────────────────────────


def print_summary(entries: list[dict]) -> None:
    n = len(entries)
    with_similar = [e for e in entries if e["num_similar_found"] > 0]
    counts = [e["num_similar_found"] for e in entries]
    non_zero = [c for c in counts if c > 0]

    response_dist: Counter[str] = Counter(e["query_company_response"] for e in entries)
    product_dist: Counter[str] = Counter(e["query_product"] for e in entries)
    issue_dist: Counter[str] = Counter(e["query_issue"] for e in entries)

    print("\n" + "=" * 65)
    print("  EVAL SET SUMMARY")
    print("=" * 65)
    print(f"  Total eval entries          : {n}")
    print(f"  With similar complaints     : {len(with_similar)}")
    print(f"  With no similar complaints  : {n - len(with_similar)}")
    if non_zero:
        print(
            f"  Similar per query  min/mean/max : "
            f"{min(non_zero)} / {sum(non_zero) / len(non_zero):.1f} / {max(non_zero)}"
        )

    print("\n  By company_response (resolution ground truth):")
    for resp, count in response_dist.most_common():
        print(f"    {resp:<45} {count}")

    print("\n  By product:")
    for product in PRODUCTS:
        print(f"    {product:<52} {product_dist[product]}")

    print("\n  Top 10 issues:")
    for issue, count in issue_dist.most_common(10):
        print(f"    {issue[:55]:<55} {count}")

    print("=" * 65)


# ── Main ──────────────────────────────────────────────────────────────────────


def run(
    input_path: Path,
    output_path: Path,
    cache_path: Path,
    num_queries: int,
    queries_per_product: int,
    skip_embedding: bool,
    skip_llm: bool,
) -> None:
    logger.info("Loading complaints from %s", input_path)
    all_complaints = load_complaints(input_path)
    logger.info("Loaded %d complaints", len(all_complaints))

    # ── Phase 1: Sample queries ──────────────────────────────────────────────
    logger.info("Sampling %d query complaints...", num_queries)
    query_complaints = sample_queries(all_complaints, queries_per_product)
    logger.info("Sampled %d queries", len(query_complaints))

    # ── Phase 2: Embeddings ──────────────────────────────────────────────────
    all_embeddings, all_ids = load_or_build_embeddings(all_complaints, cache_path, skip_embedding)
    id_to_index = {cid: i for i, cid in enumerate(all_ids)}

    # ── Phases 3 + 4: LLM verify + build entries ────────────────────────────
    groq_client = None
    if not skip_llm:
        import groq as _groq

        groq_client = _groq.Groq()

    entries: list[dict] = []

    def _save_and_exit(sig, frame):  # noqa: ANN001
        logger.info("\nInterrupted — saving %d entries so far...", len(entries))
        save_eval_set(output_path, entries)
        sys.exit(0)

    signal.signal(signal.SIGINT, _save_and_exit)

    query_ids = {str(q["complaint_id"]) for q in query_complaints}

    for i, query in enumerate(query_complaints, 1):
        candidates = find_candidates(query, all_complaints, all_embeddings, all_ids, id_to_index)
        # Exclude other queries from being labeled as "similar" to each other
        # to avoid eval set contamination
        candidates = [c for c in candidates if c["complaint_id"] not in query_ids]

        if not candidates:
            logger.debug("No candidates found for query %s", query["complaint_id"])
            entries.append(make_eval_entry(i, query, [], ""))
            continue

        if skip_llm:
            similar_ids = [
                c["complaint_id"] for c in candidates if c["similarity"] >= FALLBACK_SIM_THRESHOLD
            ]
            reasoning = f"Selected by cosine similarity >= {FALLBACK_SIM_THRESHOLD} (LLM skipped)"
        else:
            similar_ids, reasoning = verify_with_llm(groq_client, query, candidates)

            # Fallback: LLM failed or returned nothing but we have high-confidence candidates
            if not similar_ids:
                fallback = [
                    c["complaint_id"]
                    for c in candidates
                    if c["similarity"] >= FALLBACK_SIM_THRESHOLD
                ]
                if fallback:
                    similar_ids = fallback
                    reasoning = (
                        f"LLM verification failed; used cosine similarity >= "
                        f"{FALLBACK_SIM_THRESHOLD} as fallback"
                    )
                    logger.warning(
                        "Query %s: LLM failed, fell back to %d high-similarity candidates",
                        query["complaint_id"],
                        len(fallback),
                    )

            if i < len(query_complaints):
                time.sleep(0.5)

        entries.append(make_eval_entry(i, query, similar_ids, reasoning))

        if i % 10 == 0:
            logger.info(
                "Progress: %d/%d queries processed, saving checkpoint...",
                i,
                len(query_complaints),
            )
            save_eval_set(output_path, entries)

    save_eval_set(output_path, entries)
    print_summary(entries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the eval set for retrieval + resolution prediction evaluation."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/complaints_deduped.jsonl"),
        help="Deduplicated complaints JSONL (default: data/processed/complaints_deduped.jsonl).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/eval/eval_set.json"),
        help="Output eval set JSON (default: data/eval/eval_set.json).",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("data/eval/embeddings_cache.npz"),
        help="Embedding cache path (default: data/eval/embeddings_cache.npz).",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=80,
        help="Total number of eval queries to generate (default: 80).",
    )
    parser.add_argument(
        "--queries-per-product",
        type=int,
        default=16,
        help="Target queries per product category (default: 16).",
    )
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Load embeddings from cache instead of re-computing.",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM verification; use cosine similarity >= 0.6 as ground truth.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    run(
        input_path=args.input,
        output_path=args.output,
        cache_path=args.cache,
        num_queries=args.num_queries,
        queries_per_product=args.queries_per_product,
        skip_embedding=args.skip_embedding,
        skip_llm=args.skip_llm,
    )


if __name__ == "__main__":
    main()
