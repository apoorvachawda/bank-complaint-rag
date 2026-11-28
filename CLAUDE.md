# CLAUDE.md — Project Context for Claude Code

## What is this project?
A production-grade RAG system for bank complaint resolution intelligence.
Built as a portfolio project to demonstrate AI engineering skills (eval-driven development,
retrieval, observability, deployment).

Given a new bank complaint, the system retrieves similar past complaints from a 92k-record
CFPB corpus and **surfaces how they were resolved** (outcome distribution + relief fraction)
— giving a bank teller actionable precedent before working the case.

> **Reality check (eval-driven):** the headline design evolved based on measured results.
> Hybrid retrieval and cross-encoder reranking were implemented and **rejected** (both reduced
> recall; dense-only won). The LLM resolution *prediction* (Stage 3) **beat no baseline**, so
> the shipped product surfaces the retrieved evidence distribution instead of forcing a label.
> The full story is in docs/eval-results.md, KNOWN_LIMITATIONS.md, and ADR-010 through ADR-013.
> Do not "restore" hybrid/reranker/forced-prediction without re-reading those.

## Architecture
The deployed path is a single retrieval + tally step; the LLM pipeline exists but is gated off.
Real CFPB public data — 92k deduplicated complaints across 5 product categories.

- **`/analyze` (default, no LLM)** — embed the complaint (BGE-small), dense-retrieve its top-5
  neighbors from the vector index (leave-one-out), and **count their known `company_response`
  outcomes**. Returns `outcome_distribution` + `relief_fraction` + cited evidence. The relief
  fraction is a usable binary relief-vs-not signal (ROC-AUC 0.69).
- **`/predict` (gated behind `ENABLE_LLM_PREDICT`, default false)** — the 3-stage LangGraph
  pipeline: (1) **Enrich** — Groq LLM extracts structured metadata; (2) **Retrieve** — same dense
  retrieval; (3) **Predict** — Groq LLM reasons over the evidence for a single resolution label
  with confidence + cited IDs. Off by default because it beat no baseline and spends Groq's
  capped daily tokens.

Example `/analyze` output:
> 5 similar overdraft-fee complaints: 4 "Closed with explanation", 1 "Closed with monetary
> relief" → relief_fraction 0.2, with the 5 cited past complaints.

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
- **Recall@k / MRR@k**: Does retrieval find genuinely similar complaints? Measured on eval set
  v2 (pooled, unbiased — see ADR-010). Dense-only BGE/MiniLM ~0.46 recall@5; hybrid and
  reranking scored lower (negative findings).
- **Resolution prediction accuracy / macro-F1** (Stage 3, gated path): vs a majority-class
  baseline. Ground truth = actual `company_response`. Result: the LLM predictor loses to the
  baseline — documented negative finding, not a headline metric.
- **Relief-vs-not ROC-AUC** (the shipped signal): fraction of top-5 neighbors whose outcome is
  any relief, as a binary score; 0.69 on eval v2, computed by counting (no LLM).
- **Faithfulness** (PLANNED, not yet implemented): would Stage 3 cite only retrieved evidence?
  (LLM-as-judge). Deferred — see LATER.md.

## Tech Stack
- Python 3.11+, LangGraph, FastAPI (+ Gradio demo)
- Embeddings: sentence-transformers — **BGE-small-en-v1.5 is deployed** (its vectors are the
  Qdrant index); all-MiniLM-L6-v2 is a tied eval baseline (ADR-009). Vectors normalized; cosine
  distance (ADR-008).
- Vector DB: ChromaDB (local dev) / **Qdrant Cloud (prod)** — migration verified by eval parity
  (ADR-011).
- Reranker: **none** — cross-encoder reranking was tested and removed (reduced recall on this
  symmetric-similarity task; see eval-results.md / KNOWN_LIMITATIONS.md).
- LLM: **Groq `openai/gpt-oss-120b`** via its OpenAI-compatible API (not OpenAI/GPT-4o).
  Only used on the gated `/predict` path. Free tier caps at 200k tokens/day.
- Observability: Langfuse (per-stage latency + token cost)
- Deployment: Hugging Face Spaces + Qdrant Cloud (ADR-013)
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

# Run eval suite (retrieval recall@k/MRR on eval set v2)
python -m src.evaluation.run_retrieval_evals --retriever dense --eval-set data/eval/eval_set_v2.json
# Deterministic relief-vs-not signal (no LLM)
python -m src.evaluation.relief_signal_eval
# Resolution-prediction eval (gated LLM path, vs majority baseline)
python -m src.evaluation.predict_eval --retriever dense --n 100

# Start dev server
uvicorn src.api.main:app --reload --port 8000

# Lint
ruff check src/ tests/
ruff format src/ tests/

# Data pipeline
python -m src.data.download_cfpb
python -m src.data.deduplicate_corpus
python -m src.data.ingest --reset   # embed + load into ChromaDB (--reset drops the collection first)
```
