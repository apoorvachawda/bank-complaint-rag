"""The three pipeline stages as LangGraph nodes.

Nodes are factories: retrieval dependencies are injected so the pipeline is
agnostic to which retriever won the eval (and tests can inject fakes).
"""

import logging
from collections.abc import Callable

from langfuse import observe

from src.pipeline.llm import chat_json
from src.pipeline.models import (
    RESOLUTION_CATEGORIES,
    EnrichedComplaint,
    EvidenceComplaint,
    PipelineState,
    ResolutionPrediction,
)

logger = logging.getLogger(__name__)

TOP_K_EVIDENCE = 5
# Keep prompts inside Groq free-tier TPM limits; complaints longer than this
# have their key facts up front in practice.
MAX_EVIDENCE_CHARS = 1200

ENRICH_SYSTEM = """You extract structured metadata from US consumer bank complaints.
Respond with JSON: {
  "product": one of ["Checking or savings account", "Credit card or prepaid card",
                     "Money transfer, virtual currency, or money service",
                     "Mortgage", "Vehicle loan or lease"],
  "issue_summary": one sentence, the core problem,
  "key_entities": list of company names / products / fee types mentioned,
  "monetary_amount_mentioned": true if a specific dollar amount is cited,
  "severity_signals": list of escalation markers (fraud claim, legal threat,
                      regulator mention, repeated contact attempts), [] if none
}"""

PREDICT_SYSTEM = f"""You are a bank-complaint resolution analyst. Given a new complaint
and similar past complaints with their known outcomes, predict the most likely
resolution category for the new complaint.

The outcome distribution of the similar complaints is your primary evidence —
weigh it against how closely each one matches the new complaint's situation.

Respond with JSON: {{
  "predicted_resolution": exactly one of {RESOLUTION_CATEGORIES},
  "confidence": 0.0-1.0,
  "reasoning": 2-3 sentences citing the evidence pattern,
  "cited_evidence_ids": complaint_ids of the past complaints that drove the prediction
}}"""


def make_enrich_node() -> Callable[[PipelineState], dict]:
    @observe(name="stage1-enrich")
    def enrich(state: PipelineState) -> dict:
        raw = chat_json(
            system=ENRICH_SYSTEM,
            user=state.complaint_text[:6000],
            name="stage1-enrich",
        )
        return {"enrichment": EnrichedComplaint.model_validate(raw)}

    return enrich


def make_retrieve_node(
    retrieve_fn: Callable[[str, str, int], list[str]],
    get_records: Callable[[list[str]], dict[str, dict]],
    top_k: int = TOP_K_EVIDENCE,
) -> Callable[[PipelineState], dict]:
    @observe(name="stage2-retrieve")
    def retrieve(state: PipelineState) -> dict:
        ids = retrieve_fn(state.complaint_text, state.exclude_complaint_id, top_k)
        records = get_records(ids)
        evidence = [
            EvidenceComplaint(
                complaint_id=id_,
                narrative=records[id_]["document"][:MAX_EVIDENCE_CHARS],
                product=records[id_]["metadata"].get("product", ""),
                issue=records[id_]["metadata"].get("issue", ""),
                company_response=records[id_]["metadata"].get("company_response", ""),
            )
            for id_ in ids
            if id_ in records
        ]
        return {"evidence": evidence}

    return retrieve


def _format_evidence(evidence: list[EvidenceComplaint]) -> str:
    blocks = []
    for e in evidence:
        blocks.append(
            f"[{e.complaint_id}] product: {e.product} | issue: {e.issue}\n"
            f"OUTCOME: {e.company_response}\n"
            f"narrative: {e.narrative}"
        )
    return "\n\n".join(blocks)


def make_predict_node() -> Callable[[PipelineState], dict]:
    @observe(name="stage3-predict")
    def predict(state: PipelineState) -> dict:
        enrichment = state.enrichment
        user = (
            f"NEW COMPLAINT:\n{state.complaint_text[:6000]}\n\n"
            f"STRUCTURED SUMMARY: {enrichment.model_dump_json() if enrichment else 'n/a'}\n\n"
            f"SIMILAR PAST COMPLAINTS WITH KNOWN OUTCOMES:\n{_format_evidence(state.evidence)}"
        )
        raw = chat_json(system=PREDICT_SYSTEM, user=user, name="stage3-predict")

        if raw.get("predicted_resolution") not in RESOLUTION_CATEGORIES:
            logger.warning(
                "Off-schema resolution %r — falling back", raw.get("predicted_resolution")
            )
            raw["predicted_resolution"] = "Closed with explanation"
            raw["confidence"] = min(float(raw.get("confidence", 0.3)), 0.3)
        return {"prediction": ResolutionPrediction.model_validate(raw)}

    return predict
