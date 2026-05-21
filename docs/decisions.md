# Architecture Decision Records (ADRs)

Each decision follows the format: **Context → Decision → Consequences**

---

## ADR-001: Use LangGraph over plain LangChain

**Date:** 2026-05-21
**Context:** The pipeline has 4 distinct stages (classify → match → retrieve → analyze). Plain LangChain chains work but make it hard to test stages independently, add conditional routing, or trace per-stage costs.
**Decision:** Use LangGraph with each stage as a node sharing typed state.
**Consequences:** More upfront boilerplate, but each node is independently testable, Langfuse traces each stage separately, and adding/swapping stages doesn't require rewriting the pipeline.

---

## ADR-002: Start with ChromaDB, migrate to Qdrant

**Date:** 2026-05-21
**Context:** Need a vector DB for semantic search. Options considered: Pinecone (managed, expensive), Weaviate (complex), Chroma (simple, local), Qdrant (good free tier, metadata filtering, hybrid search).
**Decision:** Use Chroma for local dev speed, Qdrant for deployment. Build an abstraction layer so the swap is a config change.
**Consequences:** Slight overhead from abstraction, but demonstrates ability to evaluate and migrate infrastructure. Qdrant's metadata filtering is critical for ECN/account matching in Stage 2.

---

## ADR-003: Eval-first development

**Date:** 2026-05-21
**Context:** The original proprietary POC had no evals. Every change was "does it feel right?" which is not defensible.
**Decision:** Build the eval harness before building the retrieval pipeline. All improvements measured against the labeled test set.
**Consequences:** Slower start (need to label 80 examples first), but every subsequent decision is backed by numbers. This is the single most important differentiator from a tutorial project.

---

<!-- Add new ADRs below as we make decisions -->
