"""
Remove near-duplicate complaints from the JSONL corpus.

Two complaints are "near-duplicates" if their narratives are textually very similar,
regardless of who filed them or when. This catches template-spam complaints where
many consumers copy-paste the same boilerplate text.

Approach
────────
1. Group complaints by (product, issue) — TF-IDF within a group is meaningful
   because all complaints share vocabulary. Cross-group comparison would produce
   false matches and is prohibitively expensive at 100k records.
2. Within each group, compute pairwise cosine similarity on TF-IDF vectors.
3. Build connected components (Union-Find) over pairs above the threshold —
   A≈B and B≈C → keep only the longest of {A, B, C}.
4. Groups > 5000 complaints are processed in 5000-record batches to cap the
   O(n²) similarity matrix size.

Usage
─────
  python -m src.data.deduplicate_corpus
  python -m src.data.deduplicate_corpus --threshold 0.90 --output data/processed/complaints_deduped.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

BATCH_SIZE = 5_000


# ── Union-Find ────────────────────────────────────────────────────────────────


class UnionFind:
    def __init__(self, n: int) -> None:
        self._parent = list(range(n))
        self._rank = [0] * n

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]  # path compression
            x = self._parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1


# ── Similarity within a batch ─────────────────────────────────────────────────


def _find_duplicate_pairs_in_batch(
    narratives: list[str],
    threshold: float,
) -> list[tuple[int, int]]:
    """
    Return (i, j) index pairs whose TF-IDF cosine similarity >= threshold.
    Indices are local to this batch.
    """
    if len(narratives) < 2:
        return []

    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    try:
        tfidf = vec.fit_transform(narratives)
    except ValueError:
        # All terms filtered out (e.g. very small batch)
        return []

    sim = cosine_similarity(tfidf)
    rows, cols = np.where(sim >= threshold)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if r < c]


# ── Group deduplication ───────────────────────────────────────────────────────


def deduplicate_group(
    complaints: list[dict],
    threshold: float,
) -> tuple[set[str], list[tuple[list[str], int]]]:
    """
    Deduplicate one (product, issue) group.

    Returns:
      - ids_to_remove: complaint_ids that are redundant (shorter copy in a cluster)
      - clusters: list of (cluster_complaint_ids, cluster_size) for reporting,
                  only for clusters of size >= 2
    """
    n = len(complaints)
    uf = UnionFind(n)

    # Process in batches to cap memory use
    for batch_start in range(0, n, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n)
        batch = complaints[batch_start:batch_end]
        narratives = [c["narrative"] for c in batch]
        local_pairs = _find_duplicate_pairs_in_batch(narratives, threshold)
        for li, lj in local_pairs:
            uf.union(batch_start + li, batch_start + lj)

    # Build clusters from connected components
    from collections import defaultdict as _dd
    component_map: dict[int, list[int]] = _dd(list)
    for idx in range(n):
        component_map[uf.find(idx)].append(idx)

    ids_to_remove: set[str] = set()
    clusters: list[tuple[list[str], int]] = []

    for members in component_map.values():
        if len(members) < 2:
            continue
        # Keep the complaint with the longest narrative; remove the rest
        members.sort(key=lambda i: len(complaints[i]["narrative"]), reverse=True)
        cluster_ids = [str(complaints[i]["complaint_id"]) for i in members]
        clusters.append((cluster_ids, len(members)))
        for i in members[1:]:
            ids_to_remove.add(str(complaints[i]["complaint_id"]))

    return ids_to_remove, clusters


# ── Main deduplication pipeline ───────────────────────────────────────────────


def run(input_path: Path, output_path: Path, threshold: float) -> None:
    logger.info("Loading complaints from %s", input_path)
    complaints: list[dict] = []
    with input_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                complaints.append(json.loads(line))

    original_count = len(complaints)
    logger.info("Loaded %d complaints", original_count)

    # Group by (product, issue)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for c in complaints:
        key = (c.get("product") or "", c.get("issue") or "")
        groups[key].append(c)

    logger.info("Grouped into %d product×issue buckets", len(groups))

    all_remove: set[str] = set()
    all_clusters: list[tuple[list[str], int]] = []

    # Per-product removal counts for summary
    product_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"original": 0, "removed": 0})

    for (product, issue), group in sorted(groups.items()):
        product_stats[product]["original"] += len(group)
        if len(group) < 2:
            continue
        logger.info("Processing group: %s | %s (%d complaints)", product, issue, len(group))
        ids_to_remove, clusters = deduplicate_group(group, threshold)
        all_remove.update(ids_to_remove)
        all_clusters.extend(clusters)
        product_stats[product]["removed"] += len(ids_to_remove)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as fh:
        for c in complaints:
            if str(c["complaint_id"]) not in all_remove:
                fh.write(json.dumps(c) + "\n")
                written += 1

    _print_summary(
        original_count=original_count,
        removed_count=len(all_remove),
        remaining_count=written,
        clusters=all_clusters,
        product_stats=product_stats,
        output_path=output_path,
    )


# ── Summary ───────────────────────────────────────────────────────────────────


def _print_summary(
    original_count: int,
    removed_count: int,
    remaining_count: int,
    clusters: list[tuple[list[str], int]],
    product_stats: dict[str, dict[str, int]],
    output_path: Path,
) -> None:
    removal_rate = removed_count / original_count if original_count else 0.0

    print("\n" + "=" * 70)
    print("DEDUPLICATION SUMMARY")
    print("=" * 70)
    print(f"  Original complaints : {original_count:,}")
    print(f"  Removed             : {removed_count:,}  ({removal_rate:.1%})")
    print(f"  Remaining           : {remaining_count:,}")
    print(f"  Output written to   : {output_path}")

    # Top 10 duplicate clusters
    top_clusters = sorted(clusters, key=lambda x: x[1], reverse=True)[:10]
    if top_clusters:
        print("\n  Top 10 duplicate clusters by size:")
        print(f"  {'Copies':>6}  Template (first 100 chars of longest narrative)")
        print("  " + "─" * 66)
        for cluster_ids, size in top_clusters:
            # cluster_ids[0] is the keeper (longest), rest are removed
            print(f"  {size:>6}  IDs: {', '.join(cluster_ids[:5])}" +
                  (" ..." if len(cluster_ids) > 5 else ""))

    # Per-product breakdown
    print("\n  By product:")
    header = f"  {'Product':<54} {'Orig':>7}  {'Removed':>7}  {'Remaining':>9}  {'Rate':>6}"
    print(header)
    print("  " + "─" * 86)
    for product in sorted(product_stats):
        orig = product_stats[product]["original"]
        removed = product_stats[product]["removed"]
        remaining = orig - removed
        rate = removed / orig if orig else 0.0
        print(
            f"  {product:<54} {orig:>7,}  {removed:>7,}  {remaining:>9,}  {rate:>5.1%}"
        )

    print("=" * 70)


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove near-duplicate complaints from a JSONL corpus using TF-IDF cosine similarity."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/complaints.jsonl"),
        help="Input JSONL file (default: data/processed/complaints.jsonl).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/complaints_deduped.jsonl"),
        help="Output JSONL file (default: data/processed/complaints_deduped.jsonl).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Cosine similarity threshold above which two complaints are considered duplicates (default: 0.85).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    logger.info(
        "Starting deduplication — threshold=%.2f, input=%s, output=%s",
        args.threshold, args.input, args.output,
    )
    run(input_path=args.input, output_path=args.output, threshold=args.threshold)


if __name__ == "__main__":
    main()
