from sentence_transformers import SentenceTransformer

# BGE retrieval models are trained with an instruction prefix on the QUERY side
# only — passages are embedded bare. Omitting the prefix at search time silently
# degrades recall (see ADR-009). Symmetric models (MiniLM) use no prefix.
QUERY_PREFIXES = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
}

EMBEDDING_DIMS = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "BAAI/bge-small-en-v1.5": 384,
}

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.query_prefix = QUERY_PREFIXES.get(model_name, "")
        self.dim = EMBEDDING_DIMS[model_name]

    def embed_query(self, text: str) -> list[float]:
        # normalize_embeddings=True: these models are trained for cosine similarity,
        # and the vector store ranks by cosine — unnormalized vectors skew ranking
        return self.model.encode(self.query_prefix + text, normalize_embeddings=True).tolist()

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(
            [self.query_prefix + t for t in texts], normalize_embeddings=True
        ).tolist()

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True).tolist()
