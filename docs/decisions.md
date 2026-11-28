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

---

## ADR-008: Cosine distance + normalized embeddings (not Chroma's default L2)

**Date:** 2026-11-16
**Context:** The first ingestion created the Chroma collection with defaults, which means L2
distance. Our embedder (`all-MiniLM-L6-v2`) did not normalize vectors. Sentence-transformer
models are trained with a cosine-similarity objective — L2 ranking over unnormalized vectors
produces a subtly different nearest-neighbor ordering than the model was trained to optimize,
which would have made the dense baseline silently off-model. Caught during code review before
any baseline numbers were recorded.
**Decision:** Create the collection with `hnsw:space: cosine` AND normalize embeddings at encode
time (`normalize_embeddings=True`). Either alone is sufficient (cosine on normalized vectors
equals L2 ranking on normalized vectors), but doing both makes the intent self-documenting and
robust to either side being changed later. Since Chroma fixes the distance metric at collection
creation time, this required a full re-ingestion (`ingest.py --reset`).
**Consequences:** Baseline recall numbers are measured against the metric the model was trained
for. Ingestion also became idempotent (upsert instead of add) and batch-wise (embed + insert per
5k chunk instead of embedding all 92k upfront), so re-runs are safe and a crash loses at most one
batch.

---

## ADR-009: MiniLM as the v0 baseline embedding model, BGE-small as the measured upgrade

**Date:** 2026-11-16
**Context:** The target embedding model is BGE-small-en-v1.5 (stronger retrieval benchmarks at
the same 384 dimensions). The code currently uses all-MiniLM-L6-v2. Rather than treating this as
drift, it is the deliberate eval-driven path: establish a defensible baseline with the simpler
model first, then upgrade and measure the delta.
**Decision:** Record the dense-only baseline with MiniLM, then swap to BGE-small-en-v1.5 and
report the recall improvement in the experiment log. Note for the swap: BGE models expect a
query-side prefix ("Represent this sentence for searching relevant passages: ") at search time
but NOT at index time — omitting it silently degrades recall.
**Consequences:** Every retrieval improvement (BGE, BM25 hybrid, reranker) has a measured
before/after against the same eval set, which is the core eval-first story of this project
(ADR-003). The embedding dimension stays 384, so no vector store schema change on upgrade.

---

## ADR-010: Eval set v2 — pooled candidates to fix retriever-circular ground truth

**Date:** 2026-11-16
**Context:** First cross-retriever results on eval set v1 were nonsensical: BM25 scored 0.07
recall@5 (while its actual top-5 results were topically coherent on inspection) and BGE-small
scored 0.29 — half of MiniLM's 0.58 — contradicting every public retrieval benchmark. Root
cause: v1's candidate pairs were mined exclusively from MiniLM dense top-20 neighbors before
LLM verification. Ground truth could therefore never contain a complaint that only BM25 or BGE
retrieves; MiniLM dense won by construction. This is pooling bias, a known failure mode of IR
test collections.
**Decision:** Build eval set v2 with TREC-style pooled relevance judgments: the same 80 queries
and the same LLM-judge prompt, but candidates pooled from MiniLM-dense ∪ BGE-dense ∪ BM25
top-15 each. Two deliberate methodology fixes: the judge no longer sees per-candidate cosine
similarity (v1 showed it, anchoring the judge toward dense candidates), and the judge model is
gpt-oss-120b (v1's llama-3.3-70b was retired from Groq). v1 stays in git; both sets' numbers
are reported in eval-results.md. Alternatives considered: keeping v1 with a documented caveat
(cheap, but makes hybrid-retrieval improvements unmeasurable), or a label-free proxy metric
(same-product+issue match — weak evidence).
**Consequences:** Cross-retriever comparison becomes valid; the v1 numbers remain as the record
of the discovery. Residual caveats: labels are LLM-judged (same as v1, so methodologically
consistent), and any retriever NOT in the pool (e.g. a future ColBERT) would face the same bias
and requires re-pooling. Catching a circular evaluator is itself the strongest evidence the
eval-first process works.

---

## ADR-011: Chroma → Qdrant Cloud migration, verified by eval parity

**Date:** 2026-11-16
**Context:** ADR-002 planned Chroma for dev, Qdrant for deployment. The deployed container
cannot ship a 1.8GB local Chroma dir, and Space disks are non-persistent.
**Decision:** Migrate by copying stored vectors out of Chroma (no re-embedding) in batches of
500 with idempotent upserts and exponential-backoff retries (cloud round-trips fail
transiently; one mid-run disconnect required a resumable --start-offset). Point ids are the
numeric CFPB complaint ids; narrative lives in the payload alongside filter metadata.
**Consequences:** 92,045/92,045 points migrated. Post-migration eval: recall@5 identical to
the local backend (0.2883 vs 0.2883 on dense-BGE/v1); recall@10 differed by 0.0013 — a single
tie-break flip from float score ordering across HNSW implementations, i.e. expected variance,
not data loss. The backend is a one-env-var swap (VECTOR_BACKEND), so local eval runs stay free
while deployment reads Qdrant Cloud.

---

## ADR-012: Deterministic `/analyze` surface after the LLM predictor failed its baselines

**Date:** 2026-11-23
**Context:** The Stage 3 LLM predictor (retrieval + `gpt-oss-120b`) scored 0.70
accuracy / 0.21 macro-F1 on a held-out set — below both an always-majority
baseline (0.86 / 0.23) and a plain majority-vote over retrieved neighbor
outcomes (0.81). It added noise rather than signal (0.00 F1 on all minority
classes). Retrieval itself was fine (true outcome in top-5 evidence 93% of the
time). The resolution outcome is simply weakly determined by narrative text.
**Decision:** Do not ship a forced single-label LLM prediction as the product.
Instead measure a deterministic, LLM-free signal — the fraction of top-5
retrieved neighbors whose outcome is "any relief" — as a binary relief-vs-not
score, and gate on ROC-AUC ≥ 0.65. It scored **0.69**, so ship it: the
`POST /analyze` endpoint returns the counted neighbor outcome distribution and
relief fraction, no tokens spent. `/predict` (the LLM pipeline) stays available
and traced, but is documented as not beating baseline. LLM-based reframes are
parked in LATER.md; limitations in KNOWN_LIMITATIONS.md.
**Consequences:** The deployed demo stays functional under the Groq daily token
cap (the `/analyze` path needs no LLM). The product is more honest and more
useful (show the evidence, don't fabricate confidence), and the negative Stage 3
result plus this salvage is a stronger eval-driven-development story than a
prediction number that couldn't beat a coin flip toward the majority class.

---

## ADR-013: Deploy on Hugging Face Spaces (Gradio) + Qdrant Cloud, `/predict` disabled publicly

**Date:** 2026-11-28
**Context:** The demo needs a free, public, always-reachable home that can hold
the BGE embedding model (~130MB) plus the FastAPI app in memory and talk to the
92k-vector index. Two cost risks to contain: the vector index is 1.8GB locally
(cannot ship in the image), and the LLM path spends Groq's 200k-tokens/day free
cap — a public button hitting `/predict` would exhaust it in minutes.
**Alternatives considered:**
- *HF Spaces (chosen).* Free CPU-Basic tier (2 vCPU, 16GB RAM) fits the model in
  memory and runs CPU-only, consuming no GPU quota. Native Gradio SDK. Secrets
  are first-class ("Settings → Secrets"). Downside: free Spaces sleep after ~48h
  idle, so the first request after sleep is slow (cold load of the model).
- *Google Cloud Run.* The committed Dockerfile already targets it, but the free
  tier needs a billing card on file and scales to zero with cold starts too;
  more setup for no demo benefit. Kept as the portable fallback.
- *Fly.io / Render free tiers.* Comparable, but smaller free RAM headroom for the
  model; not clearly better than Spaces.
- *Bundling the index in the image.* Rejected — 1.8GB, and Space disk is
  non-persistent, so a managed vector DB is required regardless.
**Decision:** Deploy the Gradio app on HF Spaces (CPU Basic) reading the index
from **Qdrant Cloud** (ADR-011), not a local index. API keys (Groq, Qdrant,
Langfuse) are set as **Space secrets**, never committed — `deploy/push_to_space.py`
uploads code only and sets non-secret runtime variables (`VECTOR_BACKEND=qdrant`,
`RETRIEVER_NAME=dense`). **`/predict` is disabled in public deploys**
(`ENABLE_LLM_PREDICT` unset → false, ADR-012/gate) so the public surface is the
deterministic, token-free `/analyze`; this refines ADR-012's "stays available"
to "available, but off by default in public to protect the Groq cap."
**Consequences:** A free, shareable live URL whose demo path spends no LLM
tokens and survives the daily cap. Cold-start latency after idle sleep is the
accepted tradeoff for $0 hosting. `/predict` can be enabled behind a private
key/quota by setting `ENABLE_LLM_PREDICT=true` as a Space secret.
