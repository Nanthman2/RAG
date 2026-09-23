"""Ingère les documents de data/docs/ : découpage en chunks, embeddings, stockage vectoriel.

Usage:
    python ingest.py
"""

from config import CHUNK_OVERLAP, CHUNK_SIZE, DOCS_DIR
from rag.chunking import chunk_text
from rag.ollama_client import embed
from rag.vectorstore import new_store


def main():
    doc_paths = sorted(DOCS_DIR.glob("*.md"))
    if not doc_paths:
        print(f"Aucun document trouvé dans {DOCS_DIR}")
        return

    print(f"{len(doc_paths)} document(s) trouvé(s). Reconstruction de l'index...")
    store = new_store()

    for path in doc_paths:
        text = path.read_text(encoding="utf-8")
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"  {path.name}: {len(chunks)} chunk(s)")

        ids = [f"{path.stem}-{i}" for i in range(len(chunks))]
        embeddings = [embed(chunk) for chunk in chunks]
        sources = [path.name] * len(chunks)

        store.add(ids, embeddings, chunks, sources)

    store.save()
    print(f"\nTerminé : {store.count()} chunks indexés dans data/vector_store.pkl.")


if __name__ == "__main__":
    main()
