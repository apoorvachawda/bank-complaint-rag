# Resume Bullets

_Updated as we hit measurable milestones. Each bullet follows the format:_
_**Action verb + what you built + measurable impact**_

---

## Ready to use
_(none yet — we just started)_

## In progress
- Designed and built a production-grade RAG system for bank complaint classification and duplicate detection using LangGraph, hybrid retrieval (BM25 + dense + reranker), and structured eval harness
- _(fill in recall@k, classification accuracy, latency numbers as we measure them)_

## Template bullets (fill in numbers as we go)
- Improved duplicate detection recall@5 from __% (dense-only baseline) to __% using hybrid BM25 + dense retrieval with cross-encoder reranking
- Built eval harness with __ labeled examples measuring recall@k, classification accuracy, and faithfulness; used results to justify architecture decisions
- Reduced per-query cost by __% by routing classification to GPT-4o-mini while keeping GPT-4o for complex withdrawal analysis
- Achieved __ms p50 / __ms p95 latency across 4-stage LangGraph pipeline, tracked via Langfuse observability
- Deployed as Dockerized FastAPI service with live demo URL and CI/CD pipeline
