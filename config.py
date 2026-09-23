from pathlib import Path

BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "data" / "docs"
STORE_PATH = BASE_DIR / "data" / "vector_store.pkl"

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 3

# En dessous de ce score de similarité, on considère qu'aucun document
# pertinent n'a été trouvé et on répond "je ne sais pas".
# Calibré empiriquement : avec nomic-embed-text sur ce corpus, les chunks
# pertinents dépassent ~0.6 alors que les chunks hors-sujet tournent autour
# de 0.55-0.60 (voir README, section "garde-fous"). Seuil volontairement
# serré pour limiter le bruit dans le contexte transmis au LLM.
MIN_SIMILARITY = 0.6
