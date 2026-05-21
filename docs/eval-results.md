# Evaluation Results

## Experiment Log

| Date | Experiment | Recall@5 | Recall@10 | Classification Acc | Faithfulness | Notes |
|------|-----------|----------|-----------|-------------------|-------------|-------|
| _TBD_ | Dense-only baseline (MiniLM) | - | - | - | - | First baseline |
| _TBD_ | Dense-only (BGE-small) | - | - | - | - | Embedding upgrade |
| _TBD_ | BM25-only baseline | - | - | - | - | Lexical baseline |
| _TBD_ | Hybrid (BM25 + dense) | - | - | - | - | Combined |
| _TBD_ | Hybrid + reranker | - | - | - | - | Final retrieval |

## Eval Set Details
- **Size:** 80 labeled examples (target)
- **Source:** CFPB Consumer Complaint Database
- **Labels:** classification (complaint/non-complaint), duplicate pairs, withdrawal decision + reason
- **Labeling method:** _(document how you labeled - manual? LLM-assisted + human verified?)_

## Metric Definitions
- **Recall@k:** Of the known relevant complaints, what fraction appeared in the top-k retrieved results?
- **Classification accuracy:** Binary accuracy on complaint vs non-complaint
- **Faithfulness:** Does the withdrawal analysis cite only information present in the retrieved context? (LLM-as-judge)
