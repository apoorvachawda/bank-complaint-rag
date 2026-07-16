# CLAUDE.md — Project Context for Claude Code

## What is this project?
A production-grade RAG system for bank complaint resolution intelligence.
Built as a portfolio project to demonstrate AI engineering skills (eval-driven development,
hybrid retrieval, observability, deployment).

Given a new bank complaint, the system retrieves similar past complaints from a 92k-record
CFPB corpus and predicts the likely resolution outcome — giving a bank teller actionable
intelligence before working the case.

## Architecture
3-stage LangGraph pipeline (using real CFPB public data — 92k deduplicated complaints
across 5 product categories):

1. **Contextual Enrichment** — LLM extracts structured metadata (product, issue, severity
   signals) from the complaint text. Output is a Pydantic model used as metadata filters
   in Stage 2.
2. **Hybrid Retrieval** — BM25 + dense embeddings + metadata filtering + cross-encoder
   reranker. Finds similar past complaints in the CFPB corpus. Returns top-5 with scores.
3. **Resolution Intelligence** — LLM analyzes original complaint + retrieved similar
   complaints. Predicts likely resolution outcome (e.g. "Closed with monetary relief")
   with confidence score and cited evidence IDs.

Example Stage 3 output:
> "3 out of 5 similar complaints about overdraft fees resulted in monetary relief.
> This issue type has strong precedent for fee reversal."
> predicted_resolution: "Closed with monetary relief", confidence: 0.75

## Data Pipeline
```
complaints.jsonl (raw, 99,730 records, from CFPB API)
  → deduplicate_corpus.py (removes template spam, threshold 0.90)
  → complaints_deduped.jsonl (92,045 records — 7.7% removed)
  → build_eval_set.py (creates labeled eval set)
  → eval_set.json (80 labeled examples)
```

The raw corpus contains coordinated template campaigns (Cash App, Zelle, Navy Federal)
where thousands of consumers copy-paste identical complaint text. These are removed before
ingestion — they would dominate retrieval results. See ADR-007.

## Key Principles
- **Eval-first**: Every change must be measured against the 80-example labeled test set
- **Trace everything**: All LLM calls go through Langfuse for cost/latency tracking
- **Structured state**: LangGraph nodes communicate via typed Pydantic models, not string concatenation
- **Abstraction layers**: Vector DB and LLM calls have interfaces so implementations can be swapped

## Eval Metrics
- **Recall@k**: Does retrieval find genuinely similar complaints?
- **Resolution prediction accuracy**: Does Stage 3 correctly predict the company_response
  category? Ground truth = actual `company_response` field from CFPB data (no manual
  labeling needed).
- **Faithfulness**: Does Stage 3 cite only information from retrieved evidence? (LLM-as-judge)

## Tech Stack
- Python 3.11+, LangGraph, FastAPI
- Embeddings: sentence-transformers (BGE-small-en-v1.5)
- Vector DB: ChromaDB (dev) / Qdrant (prod)
- Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2
- LLM: GPT-4o (enrichment + resolution intelligence)
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
- src/data/ — Data acquisition and preprocessing scripts
- data/raw/ — Downloaded CFPB data (not in git)
- data/processed/ — Cleaned and deduplicated JSONL (not in git)
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
- The build backend is `setuptools.build_meta`.

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

# Data pipeline
python -m src.data.download_cfpb
python -m src.data.deduplicate_corpus
```
