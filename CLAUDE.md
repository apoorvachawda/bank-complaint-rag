# CLAUDE.md — Project Context for Claude Code

## What is this project?
A production-grade RAG system for bank complaint duplicate detection via hybrid retrieval.
Built as a portfolio project to demonstrate AI engineering skills (eval-driven development,
hybrid retrieval, observability, deployment).

## Architecture
3-stage LangGraph pipeline (using real CFPB public data — 100k complaints across 5 product
categories, 20k per category):

1. **Contextual Enrichment** — LLM extracts structured metadata (product, issue, severity signals)
   from the complaint text. Output is a Pydantic model used as metadata filters in Stage 2.
2. **Hybrid Retrieve** — BM25 + dense embeddings + metadata filtering + cross-encoder reranker.
   Finds duplicate/similar complaints in the CFPB corpus. Returns top-5 with scores.
3. **Withdrawal Analysis** — LLM takes original complaint + retrieved evidence and decides
   whether the complaint is a duplicate of an existing one. If so, recommends withdrawal
   with confidence score and the cited duplicate complaint ID.

> In production, an intake classification step would filter non-complaints before Stage 1.
> Deferred for this project — see ADR-006 in docs/decisions.md.

## Data Design Decision (ADR-004 + ADR-005 + ADR-006)
We use real CFPB Consumer Complaint Database, NOT synthetic data. The original proprietary
POC used internal customer IDs for Stage 2 matching. Since CFPB has no customer IDs (PII scrubbed),
Stage 1 was redesigned to be fully ML-driven: extract structure from text rather than look up
a database record.

Withdrawal is a **triage decision** (is this a duplicate?), not an outcome field. CFPB's
`company_response` column is an outcome recorded after resolution — using it as a withdrawal
label is a category error. Withdrawal ground truth is instead built manually in the eval set
by identifying actual duplicate complaint pairs (see docs/decisions.md ADR-005).

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
- LLM: GPT-4o (enrichment + withdrawal analysis)
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

## Dependency Management
Whenever you add a new `import` for a third-party package, you MUST:
1. Add it to `pyproject.toml` under `[project] dependencies` (or `[project.optional-dependencies] dev` for dev-only tools)
2. Run the install command immediately after:
   ```bash
   .venv/bin/pip install -e ".[dev]"
   ```
Never leave the environment out of sync with `pyproject.toml`.

Notes:
- Cross-encoder reranking is provided by `sentence-transformers` (`from sentence_transformers import CrossEncoder`) — there is no separate `cross-encoder` package on PyPI.
- The build backend is `setuptools.build_meta` (not `setuptools.backends._legacy`).

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
