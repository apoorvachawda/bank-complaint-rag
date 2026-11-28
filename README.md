# Bank Complaint Resolution Intelligence

A production-grade RAG system that, given a new bank complaint, retrieves similar
past complaints from a 92k-record CFPB corpus and **surfaces how they were
resolved** — the outcome distribution and the share that got relief — so a teller
can judge a new case against precedent. Built eval-first: every retrieval decision
is backed by a number on a labeled test set, including the decisions that said "no."

> **The honest headline:** the most interesting result in this project is a
> *negative* one. Hybrid retrieval (BM25 + dense) and cross-encoder reranking —
> the techniques you're "supposed" to add — both **reduced** recall on this task.
> The system ships the simpler dense-only retriever because that's what the eval
> said was best. See [Eval Results](#eval-results).

---

## Problem

A bank teller handling a new complaint has no cheap way to know how similar cases
were resolved. This system gives them that context before they work the case —
the `POST /analyze` endpoint returns the retrieved neighbors and tallies their
known outcomes:

> *5 similar overdraft-fee complaints: 4 "Closed with explanation", 1 "Closed
> with monetary relief" → relief likelihood 20%.*
> ```json
> { "outcome_distribution": {"Closed with explanation": 4,
>                            "Closed with monetary relief": 1},
>   "relief_fraction": 0.2,
>   "evidence": [ ... 5 cited past complaints ... ] }
> ```

It shows the evidence rather than forcing a single predicted label — because the
eval found an LLM-predicted label beats no baseline, while *counting* the
neighbors' outcomes is a usable binary relief signal (ROC-AUC 0.69). See
[Eval Results](#eval-results). Ground truth is free: CFPB publishes the actual
`company_response` for every complaint, so outcomes are measurable on the entire
corpus with no manual labeling.

## Architecture

```mermaid
flowchart LR
    Q[New complaint] --> EMB[Embed query<br/>BGE-small]
    EMB -->|query vector| VDB[(Qdrant Cloud<br/>92k BGE vectors)]
    VDB -->|top-5 neighbors| AN

    subgraph Public["POST /analyze — default, no LLM"]
        AN[Tally neighbor outcomes] --> O[outcome_distribution<br/>+ relief_fraction<br/>+ cited evidence]
    end

    VDB -.top-5 evidence.-> PP
    subgraph Gated["POST /predict — ENABLE_LLM_PREDICT only"]
        E[Stage 1 Enrich<br/>Groq LLM] -.-> PP[Stage 3 Predict<br/>Groq LLM]
    end
    Q -.-> E
    PP -.traces.-> LF[Langfuse<br/>latency + cost]
```

The **default surface is `/analyze`**: embed the complaint, pull its top-5
neighbors from the vector index, and **count their known outcomes** — no LLM.
The full 3-stage LangGraph pipeline (enrich → retrieve → predict) still exists
behind `/predict` but is **off by default** (`ENABLE_LLM_PREDICT`), since its
prediction beat no baseline and it spends Groq's capped daily tokens.

| Component | What it does | Output |
|-------|-------------|--------|
| `/analyze` (default) | Dense retrieval + count the neighbors' outcomes, leave-one-out, no LLM | `outcome_distribution` + `relief_fraction` + cited evidence |
| Stage 1 · Enrich (gated) | LLM extracts product, issue summary, severity signals | `EnrichedComplaint` (Pydantic) |
| Stage 3 · Predict (gated) | LLM reasons over the evidence's outcome distribution | `ResolutionPrediction` + cited evidence IDs |

Pipeline stages communicate via typed Pydantic state, not string concatenation.
The retrieval strategy is **injected**, so swapping retrievers (or vector
backends) is a one-line config change — which is exactly what made the
eval-driven comparison below possible.

## Eval Results

Measured on 80 labeled queries. **Two eval sets exist, and the story of why is
the point:**

### The eval caught its own bias (v1 → v2)

v1's "similar complaint" labels were mined *only* from MiniLM dense neighbors, so
MiniLM won **by construction** and every other retriever was penalized for
surfacing different-but-valid complaints (BM25 scored a nonsensical 0.07). v2
fixes this with **TREC-style pooled relevance judgments**: candidates pooled from
MiniLM + BGE + BM25, same LLM judge. All comparisons below use v2. (ADR-010.)

### Retrieval (eval set v2, recall@k · leave-one-out)

| Retriever | Recall@5 | Recall@10 | MRR@10 | Verdict |
|-----------|:--------:|:---------:|:------:|---------|
| **Dense BGE-small** | **0.456** | 0.463 | 0.630 | **deployed** (indexed in Qdrant Cloud) |
| Dense MiniLM | 0.457 | 0.493 | 0.706 | ties BGE — kept only as the baseline |
| BM25-only | 0.272 | 0.254 | 0.442 | lexical floor |
| Hybrid RRF (dense+BM25) | 0.365 | 0.372 | 0.582 | ❌ worse than dense alone |
| Hybrid + cross-encoder rerank | 0.282 | 0.306 | 0.497 | ❌ worst |

MiniLM and BGE-small are a statistical tie (Δ 0.001, noise at n=80). **BGE-small
is the deployed model** because the production index in Qdrant Cloud is built
from BGE vectors (ADR-011); MiniLM is retained only as the eval baseline.

**Why the fancy techniques lost** (full write-up in [docs/eval-results.md](docs/eval-results.md)):
- **BGE ≈ MiniLM** — a 384-dim upgrade is a wash on long, domain-specific narratives.
- **Hybrid hurt** — equal-weight RRF fused a strong leg (0.46) with a weak one
  (0.27) and landed between them; reweighting only converges back to dense-alone.
- **Reranking hurt most** — `bge-reranker-base` is trained for query→passage
  *relevance* (asymmetric), but this is narrative↔narrative *similarity*
  (symmetric). It's the wrong tool; the dense bi-encoder is trained for exactly
  this objective. Diagnosed on a 12-query probe (degrades even at 158 tokens, so
  it's not just truncation).

### Resolution prediction (Stage 3, held-out) — another honest negative

| Approach | Accuracy | Macro-F1 |
|----------|:--------:|:--------:|
| Majority class ("Closed with explanation") | **0.855** | 0.230 |
| Majority vote over retrieved evidence (no LLM) | 0.812 | — |
| Dense retrieval + Groq LLM (the system) | 0.696 | 0.207 |

Held-out by construction (n=69; the complaint predicted is excluded from its own
retrieval). **The LLM prediction layer beats neither baseline** — it loses to
both the trivial always-majority guess and a dumb vote over the retrieved
neighbors' outcomes. Yet the true outcome is in the top-5 evidence **93%** of
the time, so *retrieval* works; the outcome just isn't well-determined by
narrative content (monetary relief depends on account/company specifics not in
the text). Full analysis in [docs/eval-results.md](docs/eval-results.md).
(Groq's 200k-tokens/day free cap truncated the run at 69/80.)

**The usable salvage (no LLM):** a deterministic count of the retrieved
neighbors' outcomes predicts the binary "will this get any relief?" at
**ROC-AUC 0.69** — above the 0.65 bar, computed by counting, not inference.
This is exposed as the **`POST /analyze`** endpoint, which returns the neighbor
outcome distribution and relief fraction and keeps working when the LLM
provider is rate-limited. It's the honest product surface the Stage 3 result
pointed to: show the teller the evidence, don't force a label.
See [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) and [LATER.md](LATER.md).

## Observability

Every LLM call is traced in Langfuse with per-stage latency and token cost
(Groq pricing registered). Measured: ~2.7–3.2s end-to-end per query,
~$0.0003/query (enrich + predict).

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env            # fill in GROQ / LANGFUSE / QDRANT keys

# Data pipeline (downloads ~100k complaints, dedups to 92k, embeds + indexes)
python -m src.data.download_cfpb
python -m src.data.deduplicate_corpus
python -m src.data.ingest --reset        # embeds into local ChromaDB

# Evaluate a retriever against the labeled set
python -m src.evaluation.run_retrieval_evals --retriever dense \
    --eval-set data/eval/eval_set_v2.json

# Resolution-prediction eval (held-out, vs majority baseline)
python -m src.evaluation.predict_eval --retriever dense --n 100

# Deterministic relief-vs-not signal (no LLM, no token budget)
python -m src.evaluation.relief_signal_eval

# Serve (REST: /analyze [no-LLM, default], /predict [LLM, set ENABLE_LLM_PREDICT=true],
#        /docs; Gradio demo at /ui)
uvicorn src.api.main:app --reload --port 8000
```

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Orchestration | LangGraph | Typed multi-stage state; per-stage tracing; injectable nodes |
| Embeddings | BGE-small-en-v1.5 (MiniLM baseline) | 384-dim, CPU-friendly; measured as a tie — kept BGE as deployed |
| Vector DB | ChromaDB (dev) → Qdrant Cloud (prod) | Local speed in dev; managed + metadata filtering in prod (ADR-002, migration ADR-011) |
| Reranker | *none* (tested, removed) | Cross-encoder reduced recall on this symmetric task — negative finding |
| LLM | Groq `openai/gpt-oss-120b` | Fast hosted inference; OpenAI-compatible API; the available Groq model |
| Observability | Langfuse | Per-stage latency + token cost |
| API / Demo | FastAPI + Gradio | Auto-docs + Pydantic validation; Gradio UI for the live demo |
| Deployment | Hugging Face Spaces + Qdrant Cloud | Free CPU tier holds the embedding model in memory; index lives in Qdrant (ADR-013). Note: a free Space **sleeps after ~48h idle**, so the first request after sleep is slow (cold model load). |

## Data

[CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/):
99,730 complaints downloaded (5 product categories) → **92,045 after dedup**
(7.7% removed). The corpus contained coordinated template campaigns — the largest
was 1,859 near-identical Cash App copies — removed via TF-IDF cosine dedup within
(product, issue) groups so they don't dominate retrieval (ADR-007).

Raw data and the vector index are **not** in git (see `.gitignore`); the commands
above regenerate them.

## Documentation

- [docs/decisions.md](docs/decisions.md) — 13 ADRs, each with the alternatives and the number that justified the choice
- [docs/eval-results.md](docs/eval-results.md) — full experiment log, v1-vs-v2 bias story, reranker root-cause
- [docs/resume-bullets.md](docs/resume-bullets.md) — the measured outcomes

## License

MIT
