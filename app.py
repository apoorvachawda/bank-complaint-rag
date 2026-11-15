"""Hugging Face Spaces entrypoint: Gradio demo UI + the FastAPI REST API.

One process serves both: gr.mount_gradio_app puts the demo at /ui while
the FastAPI routes (/predict, /health, /docs) stay at the root. Runs
CPU-only — models are small enough that no GPU quota is consumed.

The BM25 index is too large for the Space repo and the disk is not
persistent, so it is pulled from a HF dataset repo at startup when
BM25_DATASET_REPO is set.
"""

import os

from dotenv import load_dotenv

load_dotenv()


def ensure_bm25_index() -> None:
    path = os.environ.get("BM25_PATH", "data/processed/bm25_index.pkl")
    repo = os.environ.get("BM25_DATASET_REPO")
    if os.path.exists(path) or not repo:
        return
    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(repo_id=repo, filename="bm25_index.pkl", repo_type="dataset")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    os.symlink(downloaded, path)


ensure_bm25_index()

import gradio as gr  # noqa: E402
import uvicorn  # noqa: E402

from src.api.main import app as fastapi_app  # noqa: E402
from src.api.main import predict as api_predict  # noqa: E402
from src.api.schemas import PredictRequest  # noqa: E402

EXAMPLE = (
    "I was charged three overdraft fees of $35 each on my checking account in a single "
    "day. The bank reordered my transactions from largest to smallest so my balance went "
    "negative earlier than it should have. Customer service refused to reverse the fees. "
    "I want these fees refunded."
)


def predict_ui(complaint_text: str):
    response = api_predict(PredictRequest(complaint_text=complaint_text))
    p = response.prediction
    evidence_rows = [
        [e.complaint_id, e.company_response, e.product, e.narrative_preview]
        for e in response.evidence
    ]
    headline = f"**{p.predicted_resolution}** (confidence {p.confidence:.0%})\n\n{p.reasoning}"
    return headline, evidence_rows, f"{response.latency_ms:.0f} ms"


with gr.Blocks(title="Bank Complaint Resolution Intelligence") as demo:
    gr.Markdown(
        "# Bank Complaint Resolution Intelligence\n"
        "Predicts the likely resolution of a bank complaint from similar past cases "
        "(92k CFPB complaints · hybrid retrieval + reranking · Groq LLM). "
        "REST API: `POST /predict` — see `/docs`."
    )
    inp = gr.Textbox(label="Complaint text", lines=8, value=EXAMPLE)
    btn = gr.Button("Predict resolution", variant="primary")
    out_pred = gr.Markdown(label="Prediction")
    out_evidence = gr.Dataframe(
        headers=["complaint_id", "outcome", "product", "narrative preview"],
        label="Similar past complaints (evidence)",
    )
    out_latency = gr.Textbox(label="Latency")
    btn.click(predict_ui, inputs=inp, outputs=[out_pred, out_evidence, out_latency])

app = gr.mount_gradio_app(fastapi_app, demo, path="/ui")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
