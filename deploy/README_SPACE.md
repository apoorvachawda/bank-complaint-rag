---
title: Bank Complaint Resolution Intelligence
emoji: 🏦
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
short_description: Predict a bank complaint's resolution from similar past CFPB cases
---

# Bank Complaint Resolution Intelligence

Predicts the likely resolution of a bank complaint by retrieving similar past
complaints from a 92k-record CFPB corpus (dense retrieval over Qdrant Cloud) and
reasoning over their known outcomes with a Groq-hosted LLM.

- **Demo UI:** this page.
- **REST API:** `POST /predict` — see `/docs`.
- **Source + engineering writeup:** https://github.com/apoorvachawda/bank-complaint-rag

Built eval-first: the retrieval design (dense-only — hybrid and reranking were
measured and rejected) is justified in the repo's `docs/eval-results.md`.

## Required Space secrets

Set these in **Settings → Variables and secrets**:

| Secret | Purpose |
|--------|---------|
| `GROQ_API_KEY` | LLM inference |
| `QDRANT_URL`, `QDRANT_API_KEY` | vector index (92k complaints) |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` | tracing (optional) |

`VECTOR_BACKEND=qdrant` and `RETRIEVER_NAME=dense` are set as Variables so the
Space reads the managed index instead of a local one.
