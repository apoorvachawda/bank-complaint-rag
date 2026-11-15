"""BM25 lexical index over complaint narratives.

Complements dense retrieval where it is weakest: products whose complaints
share generic vocabulary (fees, disputes, frozen accounts) but differ in
exact terms — company names, dollar amounts, product-specific jargon —
that embeddings blur together (see eval-results.md baseline breakdown).
"""

import pickle
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

# Lowercase alphanumerics; CFPB redaction runs ("XX/XX/XXXX") become repeated
# low-IDF tokens that BM25 naturally down-weights, so no special handling.
TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, bm25: BM25Okapi, ids: list[str]):
        self.bm25 = bm25
        self.ids = ids

    @classmethod
    def build(cls, ids: list[str], texts: list[str]) -> "BM25Index":
        return cls(BM25Okapi([tokenize(t) for t in texts]), list(ids))

    def search(self, query: str, n: int) -> list[tuple[str, float]]:
        """Return top-n (complaint_id, score) pairs, best first."""
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argpartition(scores, -n)[-n:]
        top = top[np.argsort(scores[top])[::-1]]
        return [(self.ids[i], float(scores[i])) for i in top]

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "ids": self.ids}, f)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(data["bm25"], data["ids"])
