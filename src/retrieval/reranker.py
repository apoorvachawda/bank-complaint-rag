from sentence_transformers import CrossEncoder

# Local cross-encoder (no per-call API cost, no vendor dependency).
# bge-reranker-base: strong accuracy at ~280M params, runs on laptop MPS
# and fits alongside the embedder in a small deployment container.
DEFAULT_RERANKER = "BAAI/bge-reranker-base"


class Reranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER):
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def score(self, query: str, documents: list[str]) -> list[float]:
        """Relevance score for each (query, document) pair; higher is better."""
        if not documents:
            return []
        return [float(s) for s in self.model.predict([(query, d) for d in documents])]
