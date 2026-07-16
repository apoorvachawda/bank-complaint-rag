# Resume Bullets

_Updated as we hit measurable milestones. Each bullet follows the format:_
_**Action verb + what you built + measurable impact**_

---

## Ready to use
_(none yet — pipeline implementation in progress)_

## In progress
- Designing and building a production-grade RAG system for bank complaint resolution intelligence using LangGraph, hybrid retrieval (BM25 + dense + reranker), and eval-driven development on 92k deduplicated CFPB complaints

## Template bullets (fill in numbers as we go)
- Built a production-grade RAG system for bank complaint resolution intelligence using LangGraph, hybrid retrieval (BM25 + dense + reranker), achieving __% Recall@5 and __% resolution prediction accuracy on 92k CFPB complaints
- Improved retrieval Recall@5 from __% to __% using hybrid BM25 + dense retrieval with cross-encoder reranking across 92k complaints
- Built eval harness measuring Recall@k and resolution prediction accuracy; used results to justify each retrieval architecture decision
- Achieved __ms p50 / __ms p95 latency across 3-stage LangGraph pipeline (enrich → retrieve → resolve), tracked via Langfuse observability
- Deployed as Dockerized FastAPI service with live demo URL and CI/CD pipeline
- Discovered and removed 7,685 template-spam complaints (7.7%) from CFPB corpus using TF-IDF deduplication; largest cluster had 1,859 near-identical Cash App copies that would have dominated retrieval results
