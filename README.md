# Bank Complaint RAG: Classification & Duplicate Detection

A production-grade RAG system that detects duplicate bank complaints using hybrid retrieval (BM25 + dense embeddings + cross-encoder reranking) orchestrated via LangGraph.

## Problem

Bank customer support receives complaints through multiple channels (phone, email, walk-in). Many complaints are duplicates or related to existing cases. Manual triage is slow and inconsistent. This system:

1. **Enriches** the complaint with structured metadata (product, issue, severity) extracted by LLM
2. **Retrieves** semantically similar complaints using hybrid search (BM25 + dense embeddings + reranker)
3. **Analyzes** whether the complaint is a duplicate of an existing one and recommends withdrawal if so

> In production, an intake classification step would filter non-complaints before Stage 1. Deferred for this project — see [ADR-006](docs/decisions.md).

## Architecture

_(architecture diagram will go here)_

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

# Download CFPB data
python src/data/download_cfpb.py

# Run evals
python -m pytest tests/ -v

# Start API server
uvicorn src.api.main:app --reload
```

## Eval Results

See [docs/eval-results.md](docs/eval-results.md) for detailed experiment tracking.

| Retrieval Method | Recall@5 | Recall@10 |
|-----------------|----------|-----------|
| Dense-only (MiniLM) | TBD | TBD |
| BM25-only | TBD | TBD |
| Hybrid + reranker | TBD | TBD |

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Orchestration | LangGraph | Multi-stage pipeline with typed state, independent node testing |
| Embeddings | BGE-small-en-v1.5 | Best quality/size tradeoff for local inference |
| Vector DB | Qdrant | Metadata filtering, hybrid search, free cloud tier |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | High-impact, zero-cost improvement to retrieval |
| LLM | GPT-4o | Enrichment and withdrawal analysis |
| Observability | Langfuse | Per-stage traces, cost tracking, latency monitoring |
| API | FastAPI | Async, auto-docs, Pydantic validation |

## Architecture Decisions

See [docs/decisions.md](docs/decisions.md) for full ADRs.

## Data

Uses the [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/) — a public dataset of 4M+ consumer complaints about financial products. This project downloads 100k complaints (20k per product category across 5 categories) via the public API.

## License

MIT
