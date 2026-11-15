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

from src.api.schemas import EvidenceSummary, HealthResponse, PredictRequest, PredictResponse
from src.pipeline.graph import build_pipeline
from src.pipeline.models import PipelineState
from src.retrieval.factory import make_retriever

load_dotenv()
logger = logging.getLogger(__name__)

RETRIEVER_NAME = os.environ.get("RETRIEVER_NAME", "hybrid-rerank")
VECTOR_BACKEND = os.environ.get("VECTOR_BACKEND", "chroma")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "complaints_bge")
BM25_PATH = os.environ.get("BM25_PATH", "data/processed/bm25_index.pkl")

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
    )


@observe(name="complaint-pipeline")
def _run_pipeline(complaint_text: str) -> dict:
    return _state["pipeline"].invoke(PipelineState(complaint_text=complaint_text))


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
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
