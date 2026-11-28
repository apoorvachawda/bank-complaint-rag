"""The /predict gate: off by default returns 503 and points to /analyze;
on, it reaches the pipeline. We patch the module flag and the pipeline runner
so the test needs neither a live index nor Groq."""

import importlib

from fastapi.testclient import TestClient


def _client_with_stub(monkeypatch, enable: bool):
    main = importlib.import_module("src.api.main")
    monkeypatch.setattr(main, "ENABLE_LLM_PREDICT", enable)
    # Avoid lifespan (model/index load) and the real pipeline.
    monkeypatch.setattr(main.app.router, "lifespan_context", _noop_lifespan)
    return main


def _noop_lifespan(app):
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _cm(_app):
        yield

    return _cm(app)


def test_predict_disabled_returns_503_pointing_to_analyze(monkeypatch):
    main = _client_with_stub(monkeypatch, enable=False)
    with TestClient(main.app) as client:
        resp = client.post("/predict", json={"complaint_text": "x" * 40})
    assert resp.status_code == 503
    assert "/analyze" in resp.json()["detail"]


def test_predict_enabled_reaches_pipeline(monkeypatch):
    main = _client_with_stub(monkeypatch, enable=True)

    from src.pipeline.models import (
        EnrichedComplaint,
        EvidenceComplaint,
        ResolutionPrediction,
    )

    def fake_run(text):
        return {
            "prediction": ResolutionPrediction(
                predicted_resolution="Closed with explanation",
                confidence=0.5,
                reasoning="stub",
                cited_evidence_ids=["1"],
            ),
            "enrichment": EnrichedComplaint(
                product="Mortgage",
                issue_summary="stub",
                monetary_amount_mentioned=False,
            ),
            "evidence": [
                EvidenceComplaint(complaint_id="1", narrative="n", company_response="Closed")
            ],
        }

    monkeypatch.setattr(main, "_run_pipeline", fake_run)
    with TestClient(main.app) as client:
        resp = client.post("/predict", json={"complaint_text": "x" * 40})
    assert resp.status_code == 200
    assert resp.json()["prediction"]["predicted_resolution"] == "Closed with explanation"
