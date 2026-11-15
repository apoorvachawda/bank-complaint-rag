"""LLM access: Groq (OpenAI-compatible endpoint) wrapped by Langfuse.

Routing Groq through langfuse.openai gives every call a traced generation
span with token usage and latency for free — no manual instrumentation in
the pipeline nodes. Groq is the only LLM provider in this project.
"""

import json
import os

from dotenv import load_dotenv
from langfuse.openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Single model for both stages: strong enough for structured extraction and
# evidence-based prediction; Groq's inference speed keeps p95 latency low.
# (llama-3.3-70b was retired from Groq; gpt-oss-120b is the strongest
# currently-served chat model on this account.)
DEFAULT_LLM = "openai/gpt-oss-120b"

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=GROQ_BASE_URL,
            api_key=os.environ["GROQ_API_KEY"],
        )
    return _client


@retry(wait=wait_exponential(min=2, max=30), stop=stop_after_attempt(5))
def chat_json(
    system: str,
    user: str,
    name: str,
    model: str = DEFAULT_LLM,
    temperature: float = 0.0,
) -> dict:
    """One JSON-mode chat completion; retries cover Groq free-tier rate limits."""
    response = get_client().chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        name=name,  # Langfuse generation name, e.g. "stage1-enrich"
    )
    return json.loads(response.choices[0].message.content)
