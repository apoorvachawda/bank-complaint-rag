"""Hugging Face Spaces entrypoint: Gradio demo UI + the FastAPI REST API.

One process serves both: gr.mount_gradio_app puts the demo at /ui while
the FastAPI routes (/analyze, /health, /docs) stay at the root. Runs
CPU-only — models are small enough that no GPU quota is consumed.

The demo calls /analyze (deterministic retrieval + outcome tally, no LLM),
not /predict: /predict is gated off by default (ENABLE_LLM_PREDICT) because
it spends Groq's capped daily tokens and beats no baseline (see
docs/eval-results.md). The demo therefore stays up regardless of LLM quota.

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

from src.api.main import analyze as api_analyze  # noqa: E402
from src.api.main import app as fastapi_app  # noqa: E402
from src.api.schemas import PredictRequest  # noqa: E402

EXAMPLE = (
    "I was charged three overdraft fees of $35 each on my checking account in a single "
    "day. The bank reordered my transactions from largest to smallest so my balance went "
    "negative earlier than it should have. Customer service refused to reverse the fees. "
    "I want these fees refunded."
)


def analyze_ui(complaint_text: str):
    response = api_analyze(PredictRequest(complaint_text=complaint_text))
    evidence_rows = [
        [e.complaint_id, e.company_response, e.product, e.narrative_preview]
        for e in response.evidence
    ]
    dist = ", ".join(f"{n}× {outcome}" for outcome, n in response.outcome_distribution.items())
    headline = (
        f"**Relief likelihood: {response.relief_fraction:.0%}** "
        f"({response.relief_fraction * len(response.evidence):.0f} of "
        f"{len(response.evidence)} similar complaints got monetary or non-monetary relief)\n\n"
        f"Outcome distribution of similar past complaints: {dist}"
    )
    return headline, evidence_rows, f"{response.latency_ms:.0f} ms"


with gr.Blocks(title="Bank Complaint Resolution Intelligence") as demo:
    gr.Markdown(
        "# Bank Complaint Resolution Intelligence\n"
        "Finds similar past complaints in a 92k-record CFPB corpus (dense retrieval) and "
        "shows **how they were resolved** — the outcome distribution and the share that got "
        "relief — so you can judge a new case against precedent. No forced prediction: the "
        "eval showed an LLM label beats no baseline, so this surfaces the evidence instead "
        "(see the repo's eval write-up). REST API: `POST /analyze` — see `/docs`."
    )
    inp = gr.Textbox(label="Complaint text", lines=8, value=EXAMPLE)
    btn = gr.Button("Analyze precedent", variant="primary")
    out_summary = gr.Markdown(label="Resolution evidence")
    out_evidence = gr.Dataframe(
        headers=["complaint_id", "outcome", "product", "narrative preview"],
        label="Similar past complaints (evidence)",
    )
    out_latency = gr.Textbox(label="Latency")
    btn.click(analyze_ui, inputs=inp, outputs=[out_summary, out_evidence, out_latency])

app = gr.mount_gradio_app(fastapi_app, demo, path="/ui")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
