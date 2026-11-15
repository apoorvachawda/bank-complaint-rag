# Evaluation Results

## Experiment Log

| Date | Experiment | Recall@5 | Recall@10 | Resolution Pred Acc | Faithfulness | Notes |
|------|-----------|----------|-----------|---------------------|-------------|-------|
| 2026-11-16 | Dense-only baseline (MiniLM) | 0.5825 | 0.6395 | - | - | First baseline; cosine distance, normalized embeddings (ADR-008), 80 queries |
| _TBD_ | Dense-only (BGE-small) | - | - | - | - | Embedding upgrade |
| _TBD_ | BM25-only baseline | - | - | - | - | Lexical baseline |
| _TBD_ | Hybrid (BM25 + dense) | - | - | - | - | Combined |
| _TBD_ | Hybrid + reranker | - | - | - | - | Final retrieval |

## Baseline Breakdown by Product (dense-only MiniLM, 2026-11-16)

| Product | Recall@5 | Recall@10 |
|---------|----------|-----------|
| Vehicle loan or lease | 0.8750 | 0.9125 |
| Mortgage | 0.7625 | 0.7896 |
| Checking or savings account | 0.4802 | 0.5432 |
| Credit card or prepaid card | 0.4031 | 0.4839 |
| Money transfer, virtual currency, or money service | 0.3917 | 0.4681 |

Observation: retrieval is strong where complaint narratives are domain-distinctive (vehicle
loans, mortgages) and weak where narratives share generic vocabulary across many issues
(fees, disputed transactions, frozen accounts). This is exactly the gap hybrid retrieval
(BM25 for exact-term matching) and metadata filtering are expected to close — the by-product
breakdown gives us a targeted before/after, not just an overall average.

## Eval Set Details
- **Size:** 80 labeled examples (target)
- **Source:** CFPB Consumer Complaint Database (99,730 downloaded → 92,045 after deduplication)
- **Labels:** similar complaint pairs for retrieval eval + `company_response` as resolution ground truth (see ADR-006)
- **Labeling method:** Similar pairs labeled manually; resolution ground truth derived automatically from CFPB data

## Metric Definitions
- **Recall@k:** `|retrieved top-k ∩ ground truth| / min(k, |ground truth|)`. The denominator is
  capped at k because ground-truth set sizes vary per query (1..n labeled similar complaints);
  with a raw `|ground truth|` denominator, recall@5 for a query with 8 labeled pairs would cap at
  0.625 even for a perfect retriever, making scores incomparable across queries. With the capped
  denominator a perfect top-k always scores 1.0. Retrieval is leave-one-out: the query's own
  complaint is stripped from results before scoring.
- **Resolution prediction accuracy:** Does Stage 3 correctly predict the `company_response` category (e.g. "Closed with monetary relief") based on retrieved similar complaints? Ground truth is the actual CFPB `company_response` field.
- **Faithfulness:** Does Stage 3 cite only information present in the retrieved context? Measured via LLM-as-judge.
