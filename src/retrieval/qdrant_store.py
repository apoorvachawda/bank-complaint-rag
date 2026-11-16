"""Qdrant-backed vector store, interface-compatible with VectorStore (Chroma).

Deployment backend (ADR-002): the index lives in Qdrant Cloud so the app
container ships without 500MB of local vector data and restarts don't
re-ingest. Complaint ids are numeric strings, used directly as point ids.
"""

import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tenacity import retry, stop_after_attempt, wait_exponential

from src.retrieval.vector_store import SearchResult

load_dotenv()

NARRATIVE_KEY = "narrative"


class QdrantStore:
    def __init__(self, collection_name: str, embedding_dim: int):
        self.client = QdrantClient(
            url=os.environ["QDRANT_URL"],
            api_key=os.environ["QDRANT_API_KEY"],
            timeout=30,
        )
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim

    def ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE),
            )

    # Cloud round-trips fail transiently (server disconnects, LB resets);
    # upserts are idempotent so blind retry is safe.
    @retry(wait=wait_exponential(min=2, max=30), stop=stop_after_attempt(5))
    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        points = [
            PointStruct(
                id=int(id_),
                vector=vec,
                payload={NARRATIVE_KEY: doc, **meta},
            )
            for id_, vec, doc, meta in zip(ids, vectors, documents, metadatas)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(
        self, query_vector: list[float], n: int, filters: dict | None = None
    ) -> list[SearchResult]:
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=n,
            with_payload=True,
        ).points
        results = []
        for hit in hits:
            payload = dict(hit.payload or {})
            document = payload.pop(NARRATIVE_KEY, "")
            results.append(
                SearchResult(
                    id=str(hit.id),
                    document=document,
                    metadata=payload,
                    # Qdrant returns cosine similarity (higher = better);
                    # convert to a distance so ordering semantics match Chroma
                    distance=1.0 - hit.score,
                )
            )
        return results

    def get_documents(self, ids: list[str]) -> dict[str, str]:
        return {id_: rec["document"] for id_, rec in self.get_records(ids).items()}

    def get_records(self, ids: list[str]) -> dict[str, dict]:
        points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[int(i) for i in ids],
            with_payload=True,
        )
        records = {}
        for p in points:
            payload = dict(p.payload or {})
            document = payload.pop(NARRATIVE_KEY, "")
            records[str(p.id)] = {"document": document, "metadata": payload}
        return records

    def count(self) -> int:
        return self.client.count(self.collection_name).count
