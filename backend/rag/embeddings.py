"""
AgriN RAG — Embedding Module
Uses sentence-transformers/all-MiniLM-L6-v2 (384-dim, local, no API limits).
Swap to Gemini gemini-embedding-001 for production if needed.
"""

import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("agrin.rag.embeddings")

EMBEDDING_DIM = 384
MAX_CHUNK_CHARS = 7000

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Loaded all-MiniLM-L6-v2 (384-dim, local)")
    return _model


def init_embedding_model(api_key: str = None):
    """Load the local embedding model. api_key ignored (kept for interface compat)."""
    _get_model()


def embed_text(text: str) -> list[float]:
    """Embed a single text."""
    model = _get_model()
    truncated = text[:MAX_CHUNK_CHARS]
    embedding = model.encode(truncated, normalize_embeddings=True)
    return embedding.tolist()


def embed_batch(texts: list[str], batch_size: int = 128) -> list[list[float]]:
    """Embed multiple texts in batches. Fast — no API rate limits."""
    model = _get_model()
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch = [t[:MAX_CHUNK_CHARS] for t in texts[i:i + batch_size]]
        batch_num = i // batch_size + 1
        embeddings = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.extend(embeddings.tolist())
        logger.info(f"  Embedded batch {batch_num}/{total_batches} ({len(batch)} texts)")

    return all_embeddings


def embed_query(query: str) -> list[float]:
    """Embed a search query."""
    return embed_text(query)
