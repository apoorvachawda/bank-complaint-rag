# Evaluation Results

## Experiment Log

| Date | Experiment | Recall@5 | Recall@10 | Resolution Pred Acc | Faithfulness | Notes |
|------|-----------|----------|-----------|---------------------|-------------|-------|
| _TBD_ | Dense-only baseline (MiniLM) | - | - | - | - | First baseline |
| _TBD_ | Dense-only (BGE-small) | - | - | - | - | Embedding upgrade |
| _TBD_ | BM25-only baseline | - | - | - | - | Lexical baseline |
| _TBD_ | Hybrid (BM25 + dense) | - | - | - | - | Combined |
| _TBD_ | Hybrid + reranker | - | - | - | - | Final retrieval |

## Eval Set Details
- **Size:** 80 labeled examples (target)
- **Source:** CFPB Consumer Complaint Database (99,730 downloaded → 92,045 after deduplication)
- **Labels:** similar complaint pairs for retrieval eval + `company_response` as resolution ground truth (see ADR-006)
- **Labeling method:** Similar pairs labeled manually; resolution ground truth derived automatically from CFPB data

## Metric Definitions
- **Recall@k:** Of the known similar complaints (labeled pairs), what fraction appeared in the top-k retrieved results?
- **Resolution prediction accuracy:** Does Stage 3 correctly predict the `company_response` category (e.g. "Closed with monetary relief") based on retrieved similar complaints? Ground truth is the actual CFPB `company_response` field.
- **Faithfulness:** Does Stage 3 cite only information present in the retrieved context? Measured via LLM-as-judge.
