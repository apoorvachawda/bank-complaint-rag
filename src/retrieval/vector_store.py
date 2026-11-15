from typing import TypedDict

import chromadb


class SearchResult(TypedDict):
    id: str
    document: str
    metadata: dict
    distance: float


class VectorStore:
    def __init__(self, collection_name: str, embedding_dim: int, persist_dir: str = "chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            # cosine, not Chroma's default L2: embeddings are trained for cosine
            # similarity, and the space is fixed at collection creation time
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ):
        """Insert or overwrite records — idempotent, so re-running ingestion is safe."""
        if vectors and len(vectors[0]) != self.embedding_dim:
            raise ValueError(
                f"Expected {self.embedding_dim}-dim vectors, got {len(vectors[0])} — "
                "embedding model and collection are out of sync"
            )
        self.collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=documents,
            metadatas=metadatas,
        )

    def search(
        self, query_vector: list[float], n: int, filters: dict | None = None
    ) -> list[SearchResult]:
        raw = self.collection.query(
            query_embeddings=[query_vector],  # wrap single vector for ChromaDB
            n_results=n,
            where=filters,
            include=["documents", "metadatas", "distances"],
        )

        results = []
        for i in range(len(raw["ids"][0])):  # [0] unwraps the batch dimension
            results.append(
                SearchResult(
                    id=raw["ids"][0][i],
                    document=raw["documents"][0][i],
                    metadata=raw["metadatas"][0][i],
                    distance=raw["distances"][0][i],
                )
            )
        return results

    def count(self) -> int:
        return self.collection.count()

    def delete_collection(self):
        """Reset for re-ingestion."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
