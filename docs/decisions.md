# Architecture Decision Records (ADRs)

Each decision follows the format: **Context → Decision → Consequences**

---

## ADR-001: Use LangGraph over plain LangChain

**Date:** 2026-05-21
**Context:** The pipeline has distinct stages (enrich → retrieve → analyze). Plain LangChain chains work but make it hard to test stages independently, add conditional routing, or trace per-stage costs.
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

## ADR-004: Use real CFPB data; replace customer ID lookup with contextual enrichment

**Date:** 2026-05-21
**Context:** The original POC had a "local match" stage that looked up complaints by customer
ECN/account number. CFPB public data has all PII scrubbed — no customer identifiers exist.
Two options: (A) synthesize customer IDs, or (B) adapt the pipeline to work without IDs using ML.
**Decision:** Option B. Stage 2 becomes "Contextual Enrichment": LLM extracts structured metadata
(product, issue, severity signals) from complaint text. This drives metadata filtering in Stage 3.
**Consequences:** No stage is a trivial lookup — the entire pipeline is ML-driven.

---

## ADR-006: Defer classification stage; focus on 3-stage pipeline

**Date:** 2026-05-24
**Context:** The original design included a Stage 1 LLM classifier (complaint vs non-complaint).
Three problems: (1) All CFPB records are already complaints, so we'd need synthetic non-complaints
for training/eval data. (2) LLM-generated non-complaints create distribution shift — the classifier
learns to detect "real vs synthetic text" rather than "complaint vs not." (3) Classification is
the least interesting ML problem in the pipeline; any LLM achieves 95%+ accuracy without tuning.
The project's value is in retrieval quality, eval rigor, and observability — not classification.
**Decision:** Defer classification. Pipeline is 3 stages: Contextual Enrichment → Hybrid Retrieve
→ Withdrawal Analysis. In production, an intake classifier would sit before Stage 1, but it
requires real bank data (including non-complaints) to build correctly and is outside scope.
**Consequences:** Saves ~5 hours for retrieval and eval work. Architecture is cleaner and each
remaining stage is substantive. Classification can be added later with real bank data that
includes genuine non-complaints.

---

## ADR-005: Withdrawal ground truth is built from duplicate detection, not from company_response

**Date:** 2026-05-23
**Context:** The original design derived `withdraw: True/False` from CFPB's `company_response`
field ("Closed with monetary relief" → withdraw=True, etc.). This was a category error.
`company_response` is an **outcome** recorded by the company after resolution. Withdrawal is a
**triage decision** made before processing: "this complaint is a duplicate — we don't need to
work it." These are fundamentally different concepts at different points in time.
**Decision:** Remove `withdraw` and `withdrawal_reason` from the download script entirely.
Keep `company_response` as raw context metadata. Build withdrawal ground truth manually in the
eval set (data/eval/) by identifying actual duplicate complaint pairs: two complaints with high
semantic similarity + same product + same company + close dates. Stage 4 then learns to recommend
withdrawal when it detects a duplicate, not when the company happened to offer relief.
**Consequences:** The eval set requires manual labeling of duplicate pairs (harder than reading a
column), but the resulting labels are conceptually correct and the problem is more interesting:
the system must reason about semantic similarity, not just map a string to a boolean.
