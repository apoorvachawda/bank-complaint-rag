# Evaluation Results

## Experiment Log

| Date | Experiment | Recall@5 | Recall@10 | Faithfulness | Notes |
|------|-----------|----------|-----------|-------------|-------|
| _TBD_ | Dense-only baseline (MiniLM) | - | - | - | First baseline |
| _TBD_ | Dense-only (BGE-small) | - | - | - | Embedding upgrade |
| _TBD_ | BM25-only baseline | - | - | - | Lexical baseline |
| _TBD_ | Hybrid (BM25 + dense) | - | - | - | Combined |
| _TBD_ | Hybrid + reranker | - | - | - | Final retrieval |

## Eval Set Details
- **Size:** 80 labeled examples (target)
- **Source:** CFPB Consumer Complaint Database (100k downloaded, 20k per product category)
- **Labels:** duplicate complaint pairs + withdrawal decision (derived from duplicate pairs — see ADR-005). Classification deferred — see ADR-006.
- **Labeling method:** _(document how you labeled - manual? LLM-assisted + human verified?)_

## Metric Definitions
- **Recall@k:** Of the known relevant complaints (duplicate pairs), what fraction appeared in the top-k retrieved results?
- **Faithfulness:** Does the withdrawal analysis cite only information present in the retrieved context? (LLM-as-judge)
