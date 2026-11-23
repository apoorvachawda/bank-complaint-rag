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

## Resolution Prediction (Stage 3) — a negative result worth reading

Held-out design: the complaint being predicted is excluded from its own
retrieval results; majority-class baseline computed on the train rows only.
Test sample seed=42, n=69 completed (80 requested; 11 dropped when the Groq
free-tier **200k tokens/day** cap was hit mid-run — see note below).

| Approach | Accuracy | Macro-F1 | Notes |
|----------|:--------:|:--------:|-------|
| Majority class (always "Closed with explanation") | **0.855** | 0.230 | trivial baseline |
| Majority vote over retrieved evidence outcomes | 0.812 | — | no LLM; just count the 5 neighbors' outcomes |
| **Dense retrieval + Groq LLM (the system)** | **0.696** | 0.207 | **loses to both baselines** |

**The LLM prediction layer makes things worse, not better.** It underperforms
both the trivial always-majority baseline (0.86) and a dumb majority-vote over
the retrieved neighbors' outcomes (0.81). Per-class, it gets "Closed with
explanation" right (F1 0.83) but scores **0.00 on all three minority classes**
(monetary relief n=3, non-monetary n=6, untimely n=1) — it deviates from the
evidence consensus toward minority labels and is wrong when it does.

**But retrieval is not the problem.** The true outcome appears somewhere in the
top-5 retrieved evidence **93%** of the time, and the evidence-majority-vote
alone scores 0.81. The signal is in the retrieved set; the LLM is adding
variance on top of it.

### What this means (the honest interview story)

1. **The task is base-rate-dominated.** 85% of complaints close "with
   explanation," so always guessing it is a strong 0.86 baseline. Accuracy is
   nearly useless here; macro-F1 is the honest metric and everything is weak on
   it because the minority classes have tiny support.
2. **Resolution outcome is weakly determined by narrative content.** Whether a
   complaint gets *monetary relief* depends on account specifics and company
   policy that aren't in the public narrative — so similar-looking complaints
   have mixed outcomes, and no amount of retrieval quality fixes that.
3. **The product implication:** a forced single-label prediction is the wrong
   output. The right output is to **surface the evidence outcome distribution**
   ("4 of 5 similar complaints closed with explanation; 1 with monetary relief")
   and let the teller judge — which is both more useful and more defensible than
   a prediction that can't beat a coin that always says "explanation."

### Blocker / reproducibility note

The Groq free tier caps at **200,000 tokens/day**. Building eval set v2 (80
pooled-candidate judge calls) plus this prediction run (69 × 2 stages)
exhausted the daily budget, which is why n=69 not 80. Re-running the full n=80,
or testing a reframed Stage 3 (binary relief-vs-not, or evidence-distribution
output), needs either the next-day reset or a Groq paid tier.

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
