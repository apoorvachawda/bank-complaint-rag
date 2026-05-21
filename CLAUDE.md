# CLAUDE.md — Project Context for Claude Code

## What is this project?
A production-grade RAG system for bank complaint classification and duplicate detection.
Built as a portfolio project to demonstrate AI engineering skills (eval-driven development,
hybrid retrieval, observability, deployment).

## Architecture
4-stage LangGraph pipeline:
1. **Classify** — LLM classifies input as complaint vs non-complaint
2. **Local Match** — DB lookup by customer ECN/account number
3. **Hybrid Retrieve** — BM25 + dense embeddings + cross-encoder reranker finds duplicates
4. **Withdrawal Analysis** — LLM analyzes if complaint should be withdrawn, with confidence score

## Key Principles
- **Eval-first**: Every change must be measured against the 80-example labeled test set
- **Trace everything**: All LLM calls go through Langfuse for cost/latency tracking
- **Structured state**: LangGraph nodes communicate via typed Pydantic models, not string concatenation
- **Abstraction layers**: Vector DB and LLM calls have interfaces so implementations can be swapped

## Tech Stack
- Python 3.11+, LangGraph, FastAPI
- Embeddings: sentence-transformers (BGE-small-en-v1.5)
- Vector DB: ChromaDB (dev) / Qdrant (prod)
- Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2
- LLM: GPT-4o-mini (classification) / GPT-4o (analysis)
- Observability: Langfuse
- Data: CFPB Consumer Complaint Database

## Code Style
- Use ruff for linting (config in pyproject.toml)
- Type hints on all function signatures
- Pydantic models for all data structures crossing module boundaries
- Docstrings on public functions explaining WHY, not just WHAT

## Directory Layout
- src/pipeline/ — LangGraph node definitions and graph construction
- src/retrieval/ — Embedding, BM25, reranker, vector DB abstraction
- src/evaluation/ — Eval harness, metrics, labeled test data loader
- src/api/ — FastAPI routes and request/response models
- data/raw/ — Downloaded CFPB data (not in git)
- data/processed/ — Cleaned JSONL for ingestion (not in git)
- data/eval/ — Labeled eval examples (IN git — small and critical)
- docs/ — ADRs, eval results, architecture docs
- configs/ — Model and retrieval parameter configs
- tests/ — pytest tests

## Common Commands
```bash
# Run tests
pytest tests/ -v

# Run eval suite
python -m src.evaluation.run_evals

# Start dev server
uvicorn src.api.main:app --reload --port 8000

# Lint
ruff check src/ tests/
ruff format src/ tests/
```
