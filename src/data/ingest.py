import argparse
import logging

import pandas as pd

from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

NARRATIVE_FIELD = "narrative"
ID_FIELD = "complaint_id"

# Narrative itself lives in Chroma's `documents` field; metadata stays pure
# filter material for Stage 2 metadata-filtered retrieval.
METADATA_FIELDS = [
    "product",
    "sub_product",
    "issue",
    "sub_issue",
    "company_response",
    "company",
    "date_received",
    "state",
    "submitted_via",
    "timely_response",
]

BATCH_SIZE = 5_000  # embed + upsert chunk size
COLLECTION_NAME = "complaints"
EMBEDDING_DIM = 384


def read_json(file_path: str) -> pd.DataFrame:
    """Read a JSONL file and return a DataFrame."""
    return pd.read_json(file_path, lines=True)


def build_metadata(row: pd.Series) -> dict:
    """Convert a row to metadata dict, handling nulls."""
    metadata = {}
    for field in METADATA_FIELDS:
        value = row.get(field)
        if pd.isna(value) or value is None:
            metadata[field] = ""
        else:
            metadata[field] = str(value)  # ensure everything is serializable
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/complaints_deduped.jsonl")
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate the collection first (required when the distance metric changes)",
    )
    args = parser.parse_args()

    embedder = Embedder()
    vector_store = VectorStore(collection_name=args.collection, embedding_dim=EMBEDDING_DIM)

    if args.reset:
        logger.info("Resetting collection %s", args.collection)
        vector_store.delete_collection()

    df = read_json(args.input)
    df = df.dropna(subset=[NARRATIVE_FIELD])

    # Embed per batch rather than all upfront: a crash mid-run loses at most
    # one batch, and upsert makes re-running from scratch idempotent anyway.
    for chunk_start in range(0, len(df), BATCH_SIZE):
        chunk_df = df.iloc[chunk_start : chunk_start + BATCH_SIZE]

        narratives = chunk_df[NARRATIVE_FIELD].astype(str).tolist()
        vectors = embedder.embed_batch(narratives)
        metadatas = [build_metadata(row) for _, row in chunk_df.iterrows()]
        ids = chunk_df[ID_FIELD].astype(str).tolist()

        vector_store.upsert(ids=ids, vectors=vectors, documents=narratives, metadatas=metadatas)
        logger.info("Ingested %d/%d records", chunk_start + len(chunk_df), len(df))

    logger.info("Ingestion complete. Collection count: %d", vector_store.count())


if __name__ == "__main__":
    main()
