"""FastAPI service for complaint resolution intelligence.

Models and indexes load once at startup (lifespan); /predict runs the
3-stage pipeline. Routes are sync `def` so FastAPI runs them in its
threadpool — the pipeline is CPU/IO-bound sync code.
"""

import logging
import os
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langfuse import get_client, observe

from src.api.schemas import (
    AnalyzeResponse,
    EvidenceSummary,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)
from src.evaluation.relief_signal_eval import relief_fraction
from src.pipeline.graph import build_pipeline
from src.pipeline.models import PipelineState
from src.retrieval.factory import make_retriever

load_dotenv()
logger = logging.getLogger(__name__)

# dense-only won the eval: hybrid and reranking both reduced recall on this
# symmetric-similarity task (see docs/eval-results.md). Dropping the reranker
# also keeps ~1.1GB of cross-encoder out of the deployment image.
RETRIEVER_NAME = os.environ.get("RETRIEVER_NAME", "dense")
VECTOR_BACKEND = os.environ.get("VECTOR_BACKEND", "chroma")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "complaints_bge")
BM25_PATH = os.environ.get("BM25_PATH", "data/processed/bm25_index.pkl")

# /predict is off by default: it spends Groq tokens (capped at 200k/day on the
# free tier) and, per the eval, its prediction beats no baseline anyway. The
# public demo exposes the deterministic /analyze instead. Flip to "true" only
# behind your own key/quota.
ENABLE_LLM_PREDICT = os.environ.get("ENABLE_LLM_PREDICT", "false").lower() == "true"

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading retrieval stack: %s (backend=%s)", RETRIEVER_NAME, VECTOR_BACKEND)
    stack = make_retriever(RETRIEVER_NAME, collection=COLLECTION_NAME, bm25_path=BM25_PATH)
    _state["stack"] = stack
    _state["pipeline"] = build_pipeline(stack.retrieve, stack.store.get_records)
    logger.info("Ready: %d complaints indexed", stack.store.count())
    yield
    get_client().flush()


app = FastAPI(
    title="Bank Complaint Resolution Intelligence",
    description="Predicts the likely resolution of a bank complaint from similar past cases "
    "(92k CFPB complaints, hybrid retrieval, Groq LLM).",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        retriever=RETRIEVER_NAME,
        vector_backend=VECTOR_BACKEND,
        corpus_size=_state["stack"].store.count(),
        llm_predict_enabled=ENABLE_LLM_PREDICT,
    )


@observe(name="complaint-pipeline")
def _run_pipeline(complaint_text: str) -> dict:
    return _state["pipeline"].invoke(PipelineState(complaint_text=complaint_text))


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if not ENABLE_LLM_PREDICT:
        raise HTTPException(
            status_code=503,
            detail="LLM prediction is disabled. Use POST /analyze for the deterministic "
            "evidence distribution (no LLM). See KNOWN_LIMITATIONS.md.",
        )
    start = time.perf_counter()
    try:
        final = _run_pipeline(request.complaint_text)
    except Exception:
        logger.exception("Pipeline failure")
        raise HTTPException(status_code=502, detail="Pipeline failure — see server logs")

    return PredictResponse(
        prediction=final["prediction"],
        enrichment=final["enrichment"],
        evidence=[
            EvidenceSummary(
                complaint_id=e.complaint_id,
                product=e.product,
                issue=e.issue,
                company_response=e.company_response,
                narrative_preview=e.narrative[:300],
            )
            for e in final["evidence"]
        ],
        latency_ms=(time.perf_counter() - start) * 1000,
    )


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: PredictRequest) -> AnalyzeResponse:
    """Retrieval + outcome tally, NO LLM. This is the deterministic, honest
    surface: it counts the retrieved neighbors' known outcomes instead of
    asking an LLM to force a single label (which beat no baseline). Runs even
    when the LLM provider is rate-limited, and the relief fraction is a usable
    binary signal (eval-set ROC-AUC 0.69)."""
    start = time.perf_counter()
    stack = _state["stack"]
    ids = stack.retrieve(request.complaint_text, "", 5)
    records = stack.store.get_records(ids)

    evidence, outcomes = [], []
    for id_ in ids:
        if id_ not in records:
            continue
        meta = records[id_]["metadata"]
        outcome = meta.get("company_response", "")
        outcomes.append(outcome)
        evidence.append(
            EvidenceSummary(
                complaint_id=id_,
                product=meta.get("product", ""),
                issue=meta.get("issue", ""),
                company_response=outcome,
                narrative_preview=records[id_]["document"][:300],
            )
        )

    distribution: dict[str, int] = {}
    for outcome in outcomes:
        distribution[outcome] = distribution.get(outcome, 0) + 1

    return AnalyzeResponse(
        evidence=evidence,
        outcome_distribution=distribution,
        relief_fraction=relief_fraction(outcomes),
        latency_ms=(time.perf_counter() - start) * 1000,
    )
