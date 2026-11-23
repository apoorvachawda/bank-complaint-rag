# Known Limitations

Honest list of what this system does *not* do well, with the evidence. Kept in
the repo root because an interviewer should be able to find it in ten seconds —
and because being able to state these is the point of an eval-driven project.

## 1. The LLM resolution predictor beats no baseline

On a held-out sample (n=69, dense retrieval + Groq `gpt-oss-120b`):

| Approach | Accuracy | Macro-F1 |
|----------|:--------:|:--------:|
| Always-majority ("Closed with explanation") | 0.855 | 0.230 |
| Majority vote over retrieved neighbors (no LLM) | 0.812 | — |
| Retrieval + LLM (the pipeline) | 0.696 | 0.207 |

The LLM loses to both baselines and scores 0.00 F1 on all three minority
classes. See [docs/eval-results.md](docs/eval-results.md) for the full analysis.
**Mitigation shipped:** the `/analyze` endpoint surfaces the counted neighbor
outcome distribution instead of a forced LLM label.

## 2. Resolution outcome is weakly determined by narrative text

Whether a complaint gets monetary relief depends on account specifics and
company policy not present in the public narrative. Similar-looking complaints
therefore have mixed outcomes — a ceiling no retriever or LLM can pass. The one
usable signal we found is the neighbor relief-fraction (binary relief-vs-not,
ROC-AUC 0.69), and even that is modest.

## 3. Hybrid retrieval and cross-encoder reranking both reduced recall

Equal-weight RRF (0.37) and `bge-reranker-base` reranking (0.28) both scored
below dense-only (0.46) on eval-v2. The reranker is trained for query→passage
*relevance*, but this task is narrative↔narrative *similarity* — a model-task
mismatch. Dense-only is deployed.

## 4. Small eval set (80 queries); labels are LLM-judged

Few-point metric differences are within noise. "Similar complaint" labels come
from an LLM judge over pooled candidates (TREC-style, ADR-010), not human
annotation — methodologically consistent but not a human gold standard. Relief
binary eval has only 30 positives, so its AUC wobbles ~±0.05.

## 5. Free-tier operational limits

- Groq free tier caps at **200k tokens/day**; a full LLM eval pass plus eval-set
  construction exhausts it (why the prediction eval ran 69/80). LLM experiments
  are day-rate-limited without a paid tier.
- The `/analyze` path is deliberately LLM-free so the deployed demo stays
  functional under the cap.
