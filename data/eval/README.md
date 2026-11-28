# Eval Dataset

80 labeled queries for evaluating retrieval quality. Two versions are committed;
**use v2 for any cross-retriever comparison** (v1 is biased — see below).

- `eval_set.json` — v1 (biased, kept for the record)
- `eval_set_v2.json` — v2 (pooled candidates, the fair set)

Both evaluate **retrieval** (Recall@k / MRR against labeled similar complaints).
Resolution-outcome ground truth needs no labeling here — it is the actual
`company_response` field already on every complaint in the corpus.

## v1 vs v2 (why there are two)

v1's `similar_complaint_ids` were mined **only from MiniLM dense top-20
neighbors**, then LLM-verified. That makes the labels circular: a complaint only
BM25 or BGE would retrieve can never be in the ground truth, so MiniLM wins by
construction and every other retriever is unfairly penalized (BM25 scored a
nonsensical 0.07). See ADR-010 and docs/eval-results.md.

v2 fixes this with **TREC-style pooled relevance judgments**: for the same 80
queries, candidates are pooled from **MiniLM + BGE + BM25 top-15 each**, and the
same LLM judge labels the union (with no similarity-score hint, which had
anchored v1's judge). v2 adds one field, `candidate_pool_size`.

## Labels are LLM-judged (not human gold)

`similar_complaint_ids` come from an **LLM judge** (v1: `llama-3.3-70b`, retired;
v2: `openai/gpt-oss-120b` on Groq) deciding, per candidate, whether it describes
the same type of banking problem as the query. This is methodologically
consistent across retrievers but is **not** a human-annotated gold standard —
with n=80, few-point metric differences are within noise. (See KNOWN_LIMITATIONS.md.)

## Schema

Each entry in `eval_set.json` (v1) / `eval_set_v2.json` (v2):

```json
{
  "eval_id": "eval_001",
  "query_complaint_id": "12345678",
  "query_narrative": "...",
  "query_product": "Credit card or prepaid card",
  "query_issue": "Fees or interest",
  "query_sub_issue": "...",
  "query_company": "...",
  "query_company_response": "Closed with monetary relief",
  "similar_complaint_ids": ["87654321", "11223344"],
  "similarity_reasoning": "both describe an annual-fee dispute ...",
  "num_similar_found": 7,
  "candidate_pool_size": 39            // v2 only
}
```

| Field | Description |
|-------|-------------|
| `eval_id` | Unique id for this eval example |
| `query_complaint_id` | The complaint being queried (excluded from its own results, leave-one-out) |
| `query_narrative` | Full complaint text |
| `query_product` / `query_issue` / `query_sub_issue` | CFPB taxonomy fields for the query |
| `query_company` | Company the complaint is about |
| `query_company_response` | Actual CFPB outcome — ground truth for resolution/relief metrics |
| `similar_complaint_ids` | LLM-judged similar complaints — ground truth for Recall@k / MRR |
| `similarity_reasoning` | The judge's one-line rationale |
| `num_similar_found` | `len(similar_complaint_ids)` |
| `candidate_pool_size` | **v2 only** — size of the pooled candidate set shown to the judge |

## `company_response` values (the outcome label space)

- `Closed with monetary relief` — financial remedy  ┐ counted as "relief" in
- `Closed with non-monetary relief` — corrective action, no payment  ┘ the relief signal
- `Closed with explanation` — explained, no action (the dominant class, ~85%)
- `Closed` — closed without elaboration
- `Untimely response` — bank responded late

## How to regenerate

```bash
# v1 (biased baseline set) — mines candidates from MiniLM neighbors, LLM-verifies
python -m src.data.build_eval_set \
  --input data/processed/complaints_deduped.jsonl \
  --output data/eval/eval_set.json \
  --num-queries 80 --queries-per-product 16
# (add --skip-embedding to reuse data/eval/embeddings_cache.npz)

# v2 (fair, pooled) — reuses v1's 80 queries, pools MiniLM+BGE+BM25 candidates,
# re-judges with gpt-oss-120b. Requires the ChromaDB indexes + BM25 index built.
python -m src.data.build_eval_set_v2 \
  --v1 data/eval/eval_set.json \
  --output data/eval/eval_set_v2.json \
  --pool 15
```
