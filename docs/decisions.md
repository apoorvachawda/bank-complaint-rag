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
**Consequences:** Slight overhead from abstraction, but demonstrates ability to evaluate and migrate infrastructure. Qdrant's metadata filtering is critical for metadata-driven retrieval in Stage 2.

---

## ADR-003: Eval-first development

**Date:** 2026-05-21
**Context:** The original proprietary POC had no evals. Every change was "does it feel right?" which is not defensible.
**Decision:** Build the eval harness before building the retrieval pipeline. All improvements measured against the labeled test set.
**Consequences:** Slower start (need to label examples first), but every subsequent decision is backed by numbers. This is the single most important differentiator from a tutorial project.

---

## ADR-004: Use real CFPB data; enrich complaint text instead of looking up customer records

**Date:** 2026-05-21
**Context:** The original proprietary POC had a lookup stage that retrieved customer records by ECN/account number. CFPB public data has all PII scrubbed — no customer identifiers exist. Two options: (A) synthesize customer IDs, or (B) adapt the pipeline to work without IDs using ML.
**Decision:** Option B. Stage 1 becomes "Contextual Enrichment": LLM extracts structured metadata (product, issue, severity signals) from complaint text. This output drives metadata filtering in Stage 2. The `company_response` field is kept as raw metadata and used as ground truth for resolution prediction in Stage 3 — it records the actual outcome after resolution, which is exactly what Stage 3 is trained to predict.
**Consequences:** No stage is a trivial lookup — the entire pipeline is ML-driven. `company_response` serves dual purpose: metadata context for retrieval and ground truth for eval, requiring no manual labeling for Stage 3 evaluation.

---

## ADR-005: Resolution Intelligence replaces Withdrawal Analysis as Stage 3

**Date:** 2026-07-16
**Context:** The original Stage 3 ("Withdrawal Analysis") was designed to predict whether a new complaint was a duplicate of an existing one and recommend withdrawal. After working with real CFPB data, three problems emerged: (1) True duplicates barely exist — CFPB deduplicates on their end, so each record is a unique submission from a unique person. (2) Template spam (Cash App, Zelle campaigns) looked like duplicates but were different customers copy-pasting viral text — handled better as a preprocessing step (see ADR-007), not runtime classification. (3) "Same company + same issue" pairs are not duplicates — they're different customers who each deserve individual resolution.

Meanwhile, `company_response` (which we had initially tried to misuse as a withdrawal label in an earlier design) turned out to be the perfect ground truth for a more useful task: predicting likely resolution outcomes based on patterns in similar past complaints.

**Decision:** Replace Withdrawal Analysis with Resolution Intelligence. Given a new complaint + retrieved similar complaints, Stage 3 predicts the likely resolution outcome (the `company_response` category) with a confidence score and cited evidence IDs. This is more useful to a bank teller (actionable intelligence vs. binary withdraw/keep) and more rigorously testable (ground truth available on every complaint, no manual labeling required).
**Consequences:** The eval story improves significantly — resolution prediction accuracy can be measured on the entire corpus, not just 80 manually labeled examples. The system delivers more actionable output. Template spam is handled upstream by deduplication rather than conflated with duplicate detection.

---

## ADR-006: Eval set design — similar complaint retrieval + resolution prediction

**Date:** 2026-07-16
**Context:** The original eval set design required manually labeled duplicate pairs (complaint A is a duplicate of complaint B). This proved nearly impossible with CFPB data: true duplicates are rare by design, and template spam pairs (which surface as high-similarity candidates) are different customers with the same complaint text, not duplicates. After extensive labeling effort, we had only 8 confirmed positive pairs, far below the target of 40 — and those 8 were questionable.
**Decision:** The eval set tests two independent things: (1) **Retrieval quality** via Recall@k — does the retrieval system surface genuinely similar complaints (same product, issue, and context)? Labeled similar pairs are easier to find than true duplicates because "similar" is a less strict criterion. (2) **Resolution prediction accuracy** — does Stage 3 correctly predict the `company_response` category given retrieved similar complaints? Ground truth comes directly from the CFPB data, requiring no manual labeling.
**Consequences:** No manual labeling needed for resolution prediction ground truth — the entire 92k-record corpus is implicitly labeled. Retrieval eval still requires some labeled similar pairs, but finding "similar complaints" is tractable. Eval coverage is dramatically broader.

---

## ADR-007: Corpus deduplication before vector DB ingestion

**Date:** 2026-07-16
**Context:** During eval set construction, discovered that the CFPB corpus contains mass template
campaigns — coordinated filings where many different consumers submit near-identical complaint text
(likely from viral social media posts or consumer advocacy campaigns). The worst offender was a
Cash App template with 1,859 near-identical copies. These would dominate retrieval results if left
in the corpus: a query about a Cash App issue would return 1,859 near-identical hits instead of
diverse evidence, artificially inflating similarity scores and crowding out genuinely relevant but
differently-worded complaints.
**Decision:** Built a TF-IDF + cosine similarity deduplication pipeline (threshold 0.90) that runs
between download and vector DB ingestion (`src/data/deduplicate_corpus.py`). Within each
(product, issue) group, near-duplicate clusters are identified via Union-Find over connected
components, and only the longest narrative is kept. The raw corpus is preserved for reference.
**Consequences:** Corpus reduced from 99,730 to 92,045 complaints (7.7% removal). Money transfer
category most affected (28.3% removal rate) due to Cash App and Zelle template campaigns.
Retrieval quality improves because genuine similar complaints are no longer crowded out by template
copies. Grouping by (product, issue) before TF-IDF makes the approach tractable at 100k scale
and avoids false matches across unrelated categories.
