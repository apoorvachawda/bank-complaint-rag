"""Copy the embedded corpus from local ChromaDB to Qdrant Cloud.

Reads vectors straight out of Chroma — no re-embedding, so the migrated
index is bit-identical and the post-migration eval isolates the backend
swap (recall@5 must be unchanged; any delta means a migration bug).
"""

import argparse
import logging

from src.retrieval.qdrant_store import QdrantStore
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

BATCH_SIZE = 500  # keep request payloads comfortably under Qdrant Cloud limits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="complaints_bge")
    parser.add_argument("--embedding-dim", type=int, default=384)
    parser.add_argument(
        "--start-offset",
        type=int,
        default=0,
        help="Resume after a partial run; upserts are idempotent so overlap is safe",
    )
    args = parser.parse_args()

    chroma = VectorStore(collection_name=args.collection, embedding_dim=args.embedding_dim)
    qdrant = QdrantStore(collection_name=args.collection, embedding_dim=args.embedding_dim)
    qdrant.ensure_collection()

    total = chroma.count()
    logger.info("Migrating %d points from Chroma to Qdrant Cloud", total)

    migrated = args.start_offset
    while migrated < total:
        batch = chroma.collection.get(
            limit=BATCH_SIZE,
            offset=migrated,
            include=["embeddings", "documents", "metadatas"],
        )
        if not batch["ids"]:
            break
        qdrant.upsert(
            ids=batch["ids"],
            vectors=[list(map(float, v)) for v in batch["embeddings"]],
            documents=batch["documents"],
            metadatas=batch["metadatas"],
        )
        migrated += len(batch["ids"])
        if migrated % 5000 < BATCH_SIZE:
            logger.info("Migrated %d/%d", migrated, total)

    logger.info("Done. Qdrant count: %d (expected %d)", qdrant.count(), total)


if __name__ == "__main__":
    main()
