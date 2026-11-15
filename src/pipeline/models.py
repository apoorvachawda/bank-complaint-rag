"""Typed state crossing pipeline stage boundaries (no string concatenation)."""

from pydantic import BaseModel, Field, field_validator

# The five CFPB company_response categories — Stage 3's label space.
RESOLUTION_CATEGORIES = [
    "Closed with monetary relief",
    "Closed with non-monetary relief",
    "Closed with explanation",
    "Closed",
    "Untimely response",
]


class EnrichedComplaint(BaseModel):
    """Stage 1 output: structured signals extracted from raw complaint text."""

    product: str = Field(description="Most likely CFPB product category")
    issue_summary: str = Field(description="One-sentence summary of the core issue")
    key_entities: list[str] = Field(
        default_factory=list,
        description="Company names, products, fee types mentioned",
    )
    monetary_amount_mentioned: bool = Field(
        description="Whether the complaint cites a specific dollar amount"
    )
    severity_signals: list[str] = Field(
        default_factory=list,
        description="Escalation markers: fraud claims, legal threats, regulator mentions",
    )


class EvidenceComplaint(BaseModel):
    """A retrieved similar complaint with its known outcome."""

    complaint_id: str
    narrative: str
    product: str = ""
    issue: str = ""
    company_response: str = ""


class ResolutionPrediction(BaseModel):
    """Stage 3 output: predicted outcome with cited evidence."""

    predicted_resolution: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    cited_evidence_ids: list[str] = Field(default_factory=list)

    @field_validator("cited_evidence_ids", mode="before")
    @classmethod
    def _coerce_ids_to_str(cls, v: list) -> list[str]:
        # LLMs return numeric complaint ids as ints; ids are strings everywhere else
        return [str(i) for i in v] if isinstance(v, list) else v


class PipelineState(BaseModel):
    """Shared LangGraph state; stages fill in their own fields."""

    complaint_text: str
    exclude_complaint_id: str = ""  # eval holdout: never retrieve the complaint itself
    enrichment: EnrichedComplaint | None = None
    evidence: list[EvidenceComplaint] = Field(default_factory=list)
    prediction: ResolutionPrediction | None = None
