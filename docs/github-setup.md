# GitHub Project Setup Guide

## Step 1: Create the repo on GitHub

```bash
# On your local machine after cloning this project:
gh repo create bank-complaint-rag --public --source=. --push
# OR manually: github.com > New Repository > bank-complaint-rag > Public
```

## Step 2: Create a GitHub Project (Kanban board)

1. Go to your GitHub profile → Projects → New Project
2. Choose "Board" template
3. Name it: "Bank Complaint RAG"
4. Create columns: **Backlog** | **This Week** | **In Progress** | **Done**

## Step 3: Create these issues

Copy-paste each issue below into GitHub Issues. Add the labels first:
- Labels to create: `phase-1-data`, `phase-2-baseline`, `phase-3-hybrid`, `phase-4-pipeline`, `phase-5-observability`, `phase-6-deploy`, `eval`, `infra`

---

### WEEK 1 ISSUES (move these to "This Week" column)

---

**Issue #1: Download and explore CFPB dataset**
Labels: `phase-1-data`

**Goal:** Get raw CFPB complaint data, understand the schema, select relevant fields.

Tasks:
- [ ] Write `src/data/download_cfpb.py` to fetch CFPB CSV from public API
- [ ] Explore schema: identify fields that map to our pipeline (product, narrative, complaint_id, company, date, etc.)
- [ ] Document field mapping in `docs/data-mapping.md`: CFPB field → our field (complaint_type, customer_statement, product_category, etc.)
- [ ] Filter to banking/credit card complaints only (reduce from 4M+ to manageable subset)
- [ ] Output: cleaned JSONL in `data/processed/complaints.jsonl`

Acceptance: Running the script produces a JSONL file with at least 10k complaints, each having the fields we need.

---

**Issue #2: Build eval dataset (80 labeled examples)**
Labels: `phase-1-data`, `eval`

**Goal:** Create the labeled test set that ALL future experiments are measured against.

Tasks:
- [ ] Design labeling schema (see below)
- [ ] Select 80 diverse complaints from CFPB data
- [ ] Label 40 as "has duplicate in corpus" (manually find pairs)
- [ ] Label 20 as complaint vs non-complaint edge cases
- [ ] Label 20 with expected withdrawal decision + reason
- [ ] Save as `data/eval/eval_set.json` with schema documented
- [ ] Write `src/evaluation/load_eval.py` to load and validate the eval set

Labeling schema:
```json
{
  "id": "eval_001",
  "query_complaint": "...",
  "expected_classification": "complaint",
  "expected_duplicate_ids": ["complaint_123", "complaint_456"],
  "expected_withdrawal": true,
  "expected_withdrawal_reason": "duplicate",
  "metadata": {
    "product_category": "credit_card",
    "difficulty": "hard"
  }
}
```

Acceptance: `data/eval/eval_set.json` exists with 80 entries, passes schema validation, has reasonable label distribution.

---

**Issue #3: Implement eval harness with recall@k**
Labels: `phase-1-data`, `eval`

**Goal:** Build the eval framework so we can measure every subsequent experiment.

Tasks:
- [ ] Write `src/evaluation/metrics.py` with: recall_at_k, precision_at_k, classification_accuracy, mrr (mean reciprocal rank)
- [ ] Write `src/evaluation/run_evals.py` that loads eval set, runs retrieval, computes metrics, outputs table
- [ ] Make it work with a "retriever interface" so we can swap retrieval strategies
- [ ] Add `--output` flag to save results as JSON (for `docs/eval-results.md` tracking)
- [ ] Write unit tests for metrics functions

Acceptance: `python -m src.evaluation.run_evals --retriever dummy` runs end-to-end and prints a metrics table (with zeros, since dummy retriever returns nothing).

---

**Issue #4: Dense-only retrieval baseline**
Labels: `phase-2-baseline`

**Goal:** Get the simplest possible retrieval working and measure it. This is the number we're trying to beat.

Tasks:
- [ ] Write `src/retrieval/embedder.py` — wrapper around sentence-transformers (MiniLM to start)
- [ ] Write `src/retrieval/vector_store.py` — ChromaDB abstraction with add/query interface
- [ ] Write `src/data/ingest.py` — embeds complaints JSONL and loads into Chroma
- [ ] Run eval harness against dense-only retrieval
- [ ] Record baseline numbers in `docs/eval-results.md`

Acceptance: `recall@5` and `recall@10` numbers recorded for dense-only MiniLM baseline.

---

**Issue #5: Set up dev environment and CI**
Labels: `infra`

**Goal:** Anyone (including interviewers) can clone and run the project.

Tasks:
- [ ] Verify `pip install -e ".[dev]"` works cleanly
- [ ] Add `Makefile` with common commands (install, test, lint, format, run)
- [ ] Set up pre-commit hooks (ruff format + ruff check)
- [ ] Add GitHub Actions CI: lint + test on push
- [ ] Verify README quick start works from clean clone

Acceptance: `make test` and `make lint` pass. GitHub Actions green on push.

---

### WEEK 2+ ISSUES (keep in "Backlog" column)

---

**Issue #6: BM25 retrieval baseline**
Labels: `phase-2-baseline`
Implement BM25 retrieval using rank-bm25. Run eval. Record numbers alongside dense baseline.

**Issue #7: Hybrid retrieval (BM25 + dense)**
Labels: `phase-3-hybrid`
Implement reciprocal rank fusion (RRF) to combine BM25 and dense results. Eval and compare.

**Issue #8: Cross-encoder reranker**
Labels: `phase-3-hybrid`
Add reranking step: retrieve top-20, rerank to top-5. Eval and compare to hybrid-without-reranker.

**Issue #9: Embedding model comparison**
Labels: `phase-3-hybrid`, `eval`
Swap MiniLM for BGE-small-en-v1.5. Eval. Record the improvement. This is a resume bullet.

**Issue #10: LangGraph pipeline — classify node**
Labels: `phase-4-pipeline`
Implement Stage 1: complaint classification with structured state.

**Issue #11: LangGraph pipeline — local match node**
Labels: `phase-4-pipeline`
Implement Stage 2: ECN/account number lookup.

**Issue #12: LangGraph pipeline — retrieve node**
Labels: `phase-4-pipeline`
Implement Stage 3: hybrid retrieval as a LangGraph node.

**Issue #13: LangGraph pipeline — withdrawal analysis node**
Labels: `phase-4-pipeline`
Implement Stage 4: LLM-based withdrawal recommendation.

**Issue #14: Langfuse integration**
Labels: `phase-5-observability`
Add traces to all LLM calls and retrieval. Dashboard with latency + cost per stage.

**Issue #15: FastAPI service**
Labels: `phase-6-deploy`
REST API with /classify, /search, /analyze endpoints.

**Issue #16: Docker + deployment**
Labels: `phase-6-deploy`
Dockerfile, docker-compose, deploy to Railway/Render with live URL.

**Issue #17: README polish + architecture diagram**
Labels: `phase-6-deploy`
Final README with diagram, eval table, tradeoffs. The "engineering doc" version.

---

## Step 4: Set up Claude Code

```bash
# Install Claude Code (requires Node.js 18+)
npm install -g @anthropic-ai/claude-code

# Navigate to project
cd bank-complaint-rag

# Start Claude Code — it will read CLAUDE.md automatically
claude

# Claude Code will have full context of the project from CLAUDE.md
# Use it for implementation tasks like:
# > "Implement the embedder.py wrapper for sentence-transformers"
# > "Write the eval metrics module with recall@k and MRR"
# > "Add Langfuse tracing to the classify node"
```

**How to split work between this chat and Claude Code:**
- **This chat (me):** Architecture decisions, design reviews, interview prep, "should I do X or Y?"
- **Claude Code:** Writing implementation code, debugging, running tests, file edits

Think of it as: you design here, you build there.
