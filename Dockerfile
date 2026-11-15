# Deployment image: API + models; vectors live in Qdrant Cloud.
# The BM25 pickle and eval data are NOT baked in — hybrid retrieval in
# deployment loads the BM25 index from a mounted/downloaded artifact
# (see HF Spaces setup in README).

FROM python:3.11-slim

WORKDIR /app

# Install deps first for layer caching
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e .

# Pre-download models at build time so cold starts don't pull from HF Hub
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('BAAI/bge-reranker-base')"

ENV VECTOR_BACKEND=qdrant \
    RETRIEVER_NAME=hybrid-rerank \
    COLLECTION_NAME=complaints_bge \
    PORT=7860

EXPOSE 7860

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT}"]
