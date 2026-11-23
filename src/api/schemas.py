"""Request/response models for the public API."""

from pydantic import BaseModel, Field

from src.pipeline.models import EnrichedComplaint, ResolutionPrediction


class PredictRequest(BaseModel):
    complaint_text: str = Field(min_length=20, max_length=20_000)


class EvidenceSummary(BaseModel):
    complaint_id: str
    product: str
    issue: str
    company_response: str
    narrative_preview: str


class PredictResponse(BaseModel):
    prediction: ResolutionPrediction
    enrichment: EnrichedComplaint
    evidence: list[EvidenceSummary]
    latency_ms: float


class AnalyzeResponse(BaseModel):
    """Deterministic (no-LLM) retrieval + outcome tally.

    This is the honest product surface: rather than an LLM forcing a single
    resolution label (which beat no baseline — see docs/eval-results.md), we
    return the retrieved neighbors and count their known outcomes. The relief
    fraction is a usable binary signal (ROC-AUC 0.69 on the eval set).
    """

    evidence: list[EvidenceSummary]
    outcome_distribution: dict[str, int]
    relief_fraction: float
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    retriever: str
    vector_backend: str
    corpus_size: int
