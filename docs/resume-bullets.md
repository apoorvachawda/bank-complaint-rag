# Resume Bullets

_Each bullet: action verb + what you built + measurable impact. Numbers are from
the committed eval runs (docs/eval-results.md)._

---

## Ready to use

- Built a production-grade RAG system (LangGraph, FastAPI, Qdrant, Groq) that
  surfaces resolution evidence for bank complaints from a 92k-record CFPB
  corpus, with an eval harness measuring recall@k, MRR, and resolution
  accuracy/macro-F1 on a labeled test set.

- **Caught and fixed a circular evaluation**: the first eval set mined its
  "similar complaint" labels from a single retriever's neighbors, inflating that
  retriever by construction (0.58 vs a fair 0.46 recall@5). Rebuilt it with
  TREC-style pooled relevance judgments across three retrievers, making
  cross-retriever comparison valid — the single most important correctness fix
  in the project.

- **Used the eval to reject two "best-practice" techniques with data**: hybrid
  BM25+dense fusion (0.37) and cross-encoder reranking (0.28) both *reduced*
  recall@5 below dense-only (0.46). Root-caused the reranker regression to a
  model–task mismatch (a relevance-trained cross-encoder applied to a symmetric
  similarity task) and shipped the simpler, better-measured dense retriever —
  which also cut ~1.1GB from the deployment image.

- **Measured the LLM prediction layer against two baselines and found it beat
  neither**: on a held-out, heavily-imbalanced set (85% one class), the
  retrieval+LLM predictor scored 0.70 accuracy vs 0.86 for always-guess-majority
  and 0.81 for a plain vote over retrieved neighbors — the LLM added noise.
  Retrieval itself was sound (true outcome in top-5 evidence 93% of the time);
  the finding is that CFPB resolution isn't well-determined by narrative text, so
  the right product surfaces the evidence distribution rather than forcing a label.

- Shipped an LLM-free relief-vs-not signal — the fraction of retrieved neighbors
  whose outcome was any relief — scoring **ROC-AUC 0.69** on the labeled set,
  exposed as the `/analyze` API endpoint at **zero token cost** and resilient to
  LLM rate limits (the deployed demo stays up regardless of the provider quota).

- Instrumented the full pipeline with Langfuse: ~2.7–3.2s end-to-end latency and
  ~$0.0003/query (enrich + predict), traced per stage with token-level cost.

- Discovered and removed 7,685 template-spam complaints (7.7%) from the CFPB
  corpus via TF-IDF dedup; the largest cluster was 1,859 near-identical Cash App
  copies that would otherwise dominate retrieval.

- Migrated a 92,045-vector index from local ChromaDB to Qdrant Cloud with zero
  re-embedding, verified by eval parity (recall@5 identical pre/post migration).

## Pending
- Live demo URL — fill after Hugging Face Space deploy (blocked on HF token).

## One-liner (for the top of a resume)

> Built an eval-driven RAG system (LangGraph · dense retrieval · Qdrant · Groq ·
> Langfuse) on 92k CFPB complaints; evaluated hybrid retrieval and cross-encoder
> reranking, found they didn't beat dense-only, and shipped the simpler system —
> every decision justified by numbers on a labeled set I caught and fixed for bias.
