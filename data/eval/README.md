# Eval Dataset

## Overview

80 labeled examples for evaluating the complaint resolution intelligence pipeline.
The eval set tests two independent things:

1. **Retrieval quality** — does Stage 2 surface genuinely similar complaints?
   Measured via Recall@k against labeled similar pairs.
2. **Resolution prediction accuracy** — does Stage 3 correctly predict the resolution
   outcome (`company_response` category) based on retrieved similar complaints?
   Ground truth comes directly from CFPB data — no manual labeling needed.

## Schema

Each entry in `eval_set.json`:

```json
{
  "eval_id": "eval_001",
  "query_complaint_id": "12345678",
  "query_narrative": "...",
  "query_product": "Credit card or prepaid card",
  "query_issue": "Problem with a purchase shown on your statement",
  "query_sub_issue": "...",
  "expected_resolution": "Closed with monetary relief",
  "similar_complaint_ids": ["87654321", "11223344"]
}
```

| Field | Description |
|-------|-------------|
| `eval_id` | Unique identifier for this eval example |
| `query_complaint_id` | The complaint being queried |
| `query_narrative` | Full complaint text |
| `query_product` / `query_issue` / `query_sub_issue` | Metadata used for Stage 1 enrichment |
| `expected_resolution` | Actual `company_response` from CFPB — ground truth for Stage 3 |
| `similar_complaint_ids` | Known similar complaints — ground truth for Recall@k |

## Ground Truth

**Resolution prediction:** `expected_resolution` is the actual `company_response` value
from the CFPB database. It records what the company actually did after resolving the complaint.
This is automatically available for every complaint — no manual labeling required.

Possible values:
- `Closed with monetary relief` — bank provided financial remedy
- `Closed with non-monetary relief` — bank took corrective action without payment
- `Closed with explanation` — bank explained but took no action
- `Closed` — complaint closed without elaboration
- `Untimely response` — bank responded late

**Retrieval (Recall@k):** `similar_complaint_ids` are manually labeled. A complaint is
"similar" if it describes the same type of problem at the same company — not necessarily
a true duplicate, but the kind of case a teller would want to reference when handling the query.

## How to Regenerate

```bash
python -m src.data.build_eval_set \
  --input data/processed/complaints_deduped.jsonl \
  --output data/eval/eval_set.json \
  --num-examples 80
```
