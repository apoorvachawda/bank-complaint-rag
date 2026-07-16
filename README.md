# Bank Complaint RAG: Resolution Intelligence

A production-grade RAG system that predicts likely resolution outcomes for bank complaints using hybrid retrieval (BM25 + dense embeddings + cross-encoder reranking) orchestrated via LangGraph.

## Problem

Bank customer support receives complaints through multiple channels. Triage is slow and inconsistent — agents have no visibility into how similar complaints were resolved in the past. This system:

1. **Enriches** the complaint with structured metadata (product, issue, severity) extracted by LLM
2. **Retrieves** semantically similar past complaints using hybrid search (BM25 + dense embeddings + reranker)
3. **Predicts** the likely resolution outcome based on patterns in retrieved similar complaints

Example output: *"3 out of 5 similar complaints about overdraft fees resulted in monetary relief. This issue type has strong precedent for fee reversal."* — `predicted_resolution: "Closed with monetary relief"`, `confidence: 0.75`

## Architecture

_(architecture diagram will go here)_

**3-stage LangGraph pipeline:**

| Stage | Name | What it does |
|-------|------|-------------|
| 1 | Contextual Enrichment | LLM extracts product, issue, severity from complaint text → metadata filters |
| 2 | Hybrid Retrieval | BM25 + dense embeddings + metadata filtering + cross-encoder reranker → top-5 similar complaints |
| 3 | Resolution Intelligence | LLM predicts resolution outcome from retrieved evidence, with confidence + cited IDs |

## Quick Start

```bash
# Clone and setup
git clone <repo-url>
cd bank-complaint-rag
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Fill in your API keys in .env

# Data pipeline
python -m src.data.download_cfpb           # Download 99,730 CFPB complaints
python -m src.data.deduplicate_corpus      # Remove template spam → 92,045 complaints

# Run evals
python -m src.evaluation.run_evals

# Start API server
uvicorn src.api.main:app --reload
```

## Eval Results

See [docs/eval-results.md](docs/eval-results.md) for detailed experiment tracking.

| Retrieval Method | Recall@5 | Recall@10 | Resolution Pred Acc |
|-----------------|----------|-----------|---------------------|
| Dense-only (MiniLM) | TBD | TBD | TBD |
| BM25-only | TBD | TBD | TBD |
| Hybrid + reranker | TBD | TBD | TBD |

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Orchestration | LangGraph | Multi-stage pipeline with typed state, independent node testing |
| Embeddings | BGE-small-en-v1.5 | Best quality/size tradeoff for local inference |
| Vector DB | Qdrant | Metadata filtering, hybrid search, free cloud tier |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | High-impact, zero-cost improvement to retrieval |
| LLM | GPT-4o | Enrichment and resolution intelligence |
| Observability | Langfuse | Per-stage traces, cost tracking, latency monitoring |
| API | FastAPI | Async, auto-docs, Pydantic validation |

## Data

Uses the [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/) — a public dataset of 4M+ consumer complaints about financial products.

- **Downloaded:** 99,730 complaints (20k per product category × 5 categories) via public API
- **After deduplication:** 92,045 complaints — 7,685 removed (7.7%) due to coordinated template campaigns (Cash App, Zelle, Navy Federal). Largest single cluster: 1,859 near-identical copies.
- **Deduplication method:** TF-IDF cosine similarity within (product, issue) groups, threshold 0.90

The `company_response` field (the bank's actual resolution outcome) is used as ground truth for resolution prediction — no manual labeling required for Stage 3 evaluation.

## Architecture Decisions

See [docs/decisions.md](docs/decisions.md) for full ADRs.

## License

MIT
