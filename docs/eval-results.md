# Evaluation Results

## The headline finding: eval set v1 was circular (pooling bias)

v1's ground-truth "similar complaint" pairs were mined exclusively from
**MiniLM dense top-20 candidates** (then LLM-verified). That means v1 labels can
never contain a complaint that only BM25 or BGE retrieves — so on v1, MiniLM
dense wins **by construction** and every other retriever is penalized for
surfacing different (possibly genuinely similar) complaints.

How it was caught: BM25 scored 0.07 on v1 while its actual top-5 results were
topically coherent on inspection (it even anchors correctly on company names),
and BGE-small scored half of MiniLM — contradicting every public retrieval
benchmark. Both pointed at the instrument, not the systems.

Fix: **eval set v2** (`data/eval/eval_set_v2.json`) — same 80 queries, same
LLM-judge prompt, but candidates pooled from MiniLM-dense ∪ BGE-dense ∪ BM25
top-15 (TREC-style pooled relevance judgments). v2 also stops showing the judge
per-candidate cosine scores (an anchor toward dense candidates). See ADR-010.

Retriever comparisons below are valid **within a table only**. v1 numbers are
kept as the record of the discovery, not for cross-retriever conclusions.

## Experiment Log — eval set v1 (biased toward MiniLM; see above)

| Date | Experiment | Recall@5 | Recall@10 | MRR@10 | Notes |
|------|-----------|----------|-----------|--------|-------|
| 2026-11-16 | Dense MiniLM | 0.5825 | 0.6395 | 0.7559 | Inflated by construction: labels mined from MiniLM candidates |
| 2026-11-16 | Dense BGE-small (query prefix) | 0.2883 | 0.2439 | 0.5357 | Penalized by v1 label bias |
| 2026-11-16 | Dense BGE-small (no prefix) | 0.2658 | 0.2800 | 0.5190 | Prefix is not the story — bias is |
| 2026-11-16 | BM25-only | 0.0688 | 0.0570 | 0.1377 | Penalized by v1 label bias; qualitative top-5 looks sound |

## Experiment Log — eval set v2 (pooled candidates, fair comparison)

| Date | Experiment | Recall@5 | Recall@10 | MRR@10 | Money-transfer R@5 | Notes |
|------|-----------|----------|-----------|--------|--------------------|-------|
| 2026-11-17 | Dense MiniLM | 0.4573 | 0.4929 | 0.7059 | 0.4594 | Was 0.58 on biased v1 — drop confirms the bias |
| 2026-11-17 | Dense BGE-small | 0.4556 | 0.4625 | 0.6295 | 0.4375 | **Ties MiniLM** (Δ 0.002, within noise) — negative finding for the upgrade |
| 2026-11-17 | BM25-only | 0.2723 | 0.2539 | 0.4415 | 0.2031 | 4× its biased-v1 score (0.07); strong on Mortgage (0.48) |
| 2026-11-17 | Hybrid RRF (BGE+BM25, equal weight) | 0.3646 | 0.3723 | 0.5815 | 0.3375 | **Worse than dense alone** — equal RRF drags the strong leg down |
| 2026-11-22 | Hybrid + BGE reranker | 0.2823 | 0.3055 | 0.4967 | 0.3031 | **Worst of all** — cross-encoder is the wrong tool here (see below) |

**Winner: dense-only (MiniLM or BGE-small, tied at ~0.46 recall@5).** The two
"sophisticated" additions — hybrid fusion and cross-encoder reranking — both
*reduced* recall. Deployed retriever is dense-only BGE-small (already migrated
to Qdrant; ties MiniLM, and dropping the reranker removes ~1.1GB from the
deployment image). With 80 queries, few-point deltas are noise — reported as
ties, not wins.

### Why the fancy techniques lost (the interesting part)

- **BGE ties MiniLM — the embedding upgrade is a wash.** CFPB narratives are
  long and domain-specific; a 384-dim general-purpose model captures about as
  much either way. Honest negative finding.
- **Hybrid RRF hurt.** Equal-weight fusion of a 0.27 leg (BM25) with a 0.46 leg
  (dense) lands at 0.36 — exactly between, as RRF predicts for unequal legs.
  Weighted fusion is implemented (`hybrid_retriever(dense_weight=...)`) so the
  strong leg can be favored; mathematically, as `dense_weight → ∞` the fusion
  converges to dense-alone, so the best weighted hybrid can do here is *tie*
  dense, never beat it — the full sweep was deprioritized in favor of the
  resolution-prediction eval. Net: fusion adds latency and a second index for
  no measured gain on this task.
- **The cross-encoder reranker was the biggest surprise — it hurt the most
  (0.28), even on short inputs.** Diagnosed on 12 queries: dense@5 0.48 →
  rerank@5 0.28, degrading even at 158–255 tokens (well under the 512 limit),
  so truncation of long pairs is only a secondary effect. Root cause is a
  **model–task mismatch**: `bge-reranker-base` is trained for query→passage
  *relevance* (asymmetric: short query, long answer passage), but this task is
  narrative↔narrative *similarity* (symmetric: long complaint ↔ similar long
  complaint). The reranker's "is this passage relevant to this question" signal
  does not match "do these two complaints describe the same problem." The dense
  bi-encoders are trained with a symmetric similarity objective — the right
  tool. A similarity-tuned reranker could help, but that is out of scope; the
  measured call is to ship dense-only.

**The headline story isn't "I built hybrid + reranking." It's "I built an eval
that caught its own circularity (v1→v2), then used the fair eval to show the
sophisticated retrieval techniques didn't fit this symmetric-similarity task,
and shipped the simpler, better-measured system."**

## Resolution Prediction (Stage 3)

| Date | Config | Accuracy | Macro-F1 | Baseline Acc | Baseline Macro-F1 | N |
|------|--------|----------|----------|--------------|--------------------|---|
| _pending_ | | | | | | |

Held-out design: the complaint being predicted is excluded from its own
retrieval results; majority-class baseline computed on the corpus excluding
the test split. Macro-F1 matters because resolutions are heavily imbalanced
("Closed with explanation" dominates).

## Eval Set Details
- **Size:** 80 labeled queries (same queries in v1 and v2)
- **Source:** CFPB Consumer Complaint Database (99,730 downloaded → 92,045 after deduplication)
- **v1 labels:** candidates = MiniLM dense top-20, LLM-verified (llama-3.3-70b) — biased, kept for the record
- **v2 labels:** candidates = MiniLM ∪ BGE ∪ BM25 top-15 each, LLM-verified (gpt-oss-120b), no similarity hints shown to the judge
- **Resolution ground truth:** actual `company_response` field — no manual labeling

## Metric Definitions
- **Recall@k:** `|retrieved top-k ∩ ground truth| / min(k, |ground truth|)`. The denominator is
  capped at k because ground-truth set sizes vary per query (1..n labeled similar complaints);
  with a raw `|ground truth|` denominator, recall@5 for a query with 8 labeled pairs would cap at
  0.625 even for a perfect retriever, making scores incomparable across queries. With the capped
  denominator a perfect top-k always scores 1.0. Retrieval is leave-one-out: the query's own
  complaint is stripped from results before scoring.
- **MRR@10:** mean reciprocal rank of the first relevant result within the top 10; 0 when none.
- **Resolution prediction accuracy / macro-F1:** Does Stage 3 correctly predict the
  `company_response` category? Ground truth is the actual CFPB field. Macro-F1 guards against
  the dominant class inflating accuracy.
- **Faithfulness:** Does Stage 3 cite only information present in the retrieved context? (LLM-as-judge; planned)
