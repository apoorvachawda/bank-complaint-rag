# Later / Deferred Experiments

Ideas worth trying that are out of scope for now — mostly because they need LLM
token budget beyond the Groq free-tier daily cap (see KNOWN_LIMITATIONS.md §5),
or because the deterministic `/analyze` surface already covers the near-term
need. Not committed to; parked here so the decision trail is explicit.

## LLM-based Stage 3 reframes (need token budget)

1. **Evidence-distribution-via-LLM.** Have the LLM summarize the retrieved
   neighbors' outcome distribution in natural language + a calibrated confidence,
   rather than forcing a single class. Compare its calibration (Brier / ECE)
   against the deterministic `/analyze` counts — only worth it if the LLM adds
   genuine nuance over counting, which the current evidence doubts.

2. **Binary relief-vs-not via LLM.** Prompt the LLM for just "any relief likely?
   yes/no + confidence" instead of 5-way classification, and compare ROC-AUC
   against the deterministic neighbor relief-fraction baseline (0.69, already
   measured). The LLM must *beat 0.69* to justify its cost; if it can't, the
   counting approach wins.

3. **Full n=80 (or larger) prediction eval.** The committed run is n=69 (Groq
   daily cap). Re-run at n=80+ after a reset or on a paid tier for a tighter
   number. Low priority — the conclusion (loses to baseline) is unlikely to flip.

## Retrieval

4. **Similarity-tuned reranker.** The relevance-trained cross-encoder hurt
   (KNOWN_LIMITATIONS §3). A reranker/bi-encoder fine-tuned for symmetric
   text-similarity (or a domain-adapted embedding) might actually help — the only
   reranking avenue not yet closed off.

5. **Metadata-filtered retrieval.** Restrict candidates by product/issue before
   dense search. Could sharpen the weak categories; needs the Stage-1 enrichment
   output wired into the Stage-2 filter (plumbing exists, not yet evaluated).

## Faithfulness

6. **LLM-as-judge faithfulness metric.** Does Stage 3 cite only retrieved
   evidence? Defined in docs but not implemented; needs token budget.
