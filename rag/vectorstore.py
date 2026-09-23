"""Index vectoriel maison : embeddings en mémoire (numpy), persistés sur disque en pickle.

Volumétrie visée : quelques centaines de chunks. À cette échelle, une recherche par
similarité cosinus en force brute (produit matriciel) est instantanée et évite toute
dépendance native (contrairement à chromadb/faiss, qui nécessitent un compilateur C++
sur Windows).
"""

import pickle
from dataclasses import dataclass, field

import numpy as np

from config import STORE_PATH


@dataclass
class VectorStore:
    ids: list[str] = field(default_factory=list)
    embeddings: list[list[float]] = field(default_factory=list)
    documents: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)

    def count(self) -> int:
        return len(self.ids)

    def add(self, ids, embeddings, documents, sources):
        self.ids.extend(ids)
        self.embeddings.extend(embeddings)
        self.documents.extend(documents)
        self.sources.extend(sources)

    def save(self):
        STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(STORE_PATH, "wb") as f:
            pickle.dump(self, f)

    def query(self, query_embedding, top_k: int) -> list[dict]:
        if not self.embeddings:
            return []

        matrix = np.array(self.embeddings)
        query_vec = np.array(query_embedding)

        # Similarité cosinus = produit scalaire des vecteurs normalisés.
        matrix_norms = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
        query_norm = query_vec / np.linalg.norm(query_vec)
        similarities = matrix_norms @ query_norm

        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [
            {
                "text": self.documents[i],
                "source": self.sources[i],
                "similarity": float(similarities[i]),
            }
            for i in top_indices
        ]


def get_store() -> VectorStore:
    if STORE_PATH.exists():
        with open(STORE_PATH, "rb") as f:
            return pickle.load(f)
    return VectorStore()


def new_store() -> VectorStore:
    return VectorStore()
