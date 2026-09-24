"""
AgriN RAG — Vector Store (ChromaDB)
Local-first with metadata filtering. Swap to Firestore vector search later.
"""

import os
import json
import logging
import chromadb
from chromadb.config import Settings

from .embeddings import embed_batch, embed_query

logger = logging.getLogger("agrin.rag.store")

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "data", "chromadb")
COLLECTION_NAME = "agrin_rag"

_client = None
_collection = None


def get_collection():
    """Get or create the ChromaDB collection."""
    global _client, _collection
    if _collection is not None:
        return _collection

    os.makedirs(CHROMA_DIR, exist_ok=True)
    _client = chromadb.PersistentClient(path=CHROMA_DIR)
    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(f"ChromaDB collection '{COLLECTION_NAME}' ready ({_collection.count()} docs)")
    return _collection


def build_index(chunks: list[dict], force_rebuild: bool = False):
    """
    Embed all chunks and store in ChromaDB.
    Skips if collection already populated (unless force_rebuild).
    """
    collection = get_collection()

    if collection.count() > 0 and not force_rebuild:
        logger.info(f"Index already built ({collection.count()} docs). Use force_rebuild=True to rebuild.")
        return

    if force_rebuild and collection.count() > 0:
        logger.info("Force rebuilding — clearing existing index")
        _client.delete_collection(COLLECTION_NAME)
        collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    logger.info(f"Building index for {len(chunks)} chunks...")

    # Extract texts for embedding
    texts = [c["text"] for c in chunks]
    embeddings = embed_batch(texts)

    # Prepare for ChromaDB
    ids = [c["id"] for c in chunks]
    documents = texts
    metadatas = []
    for c in chunks:
        meta = {
            "crop": c["metadata"]["crop"],
            "topic": c["metadata"]["topic"],
            "source": c["metadata"]["source"],
            "source_file": c["metadata"]["source_file"],
            "header": c.get("header", ""),
            # ChromaDB metadata values must be str/int/float/bool
            "diseases": json.dumps(c["metadata"].get("diseases", [])),
        }
        metadatas.append(meta)

    # Insert in batches (ChromaDB limit is ~5000 per add)
    batch_size = 500
    for i in range(0, len(ids), batch_size):
        end = min(i + batch_size, len(ids))
        collection.add(
            ids=ids[i:end],
            embeddings=embeddings[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end],
        )
        logger.info(f"  Indexed {end}/{len(ids)} chunks")

    logger.info(f"Index built: {collection.count()} documents")


def search(query: str, crop: str = None, topic: str = None,
           disease: str = None, top_k: int = 5) -> list[dict]:
    """
    Filtered vector search.
    
    1. Build metadata filter (crop, topic, disease)
    2. Embed query with RETRIEVAL_QUERY task type
    3. Return top-k results with scores
    """
    collection = get_collection()
    if collection.count() == 0:
        logger.warning("Empty index — no results")
        return []

    # Build where filter
    where_clauses = []
    if crop and crop != "general":
        where_clauses.append({"crop": {"$eq": crop}})
    if topic:
        where_clauses.append({"topic": {"$eq": topic}})

    where_filter = None
    if len(where_clauses) == 1:
        where_filter = where_clauses[0]
    elif len(where_clauses) > 1:
        where_filter = {"$and": where_clauses}

    # Disease filtering is trickier — stored as JSON string
    # We'll do post-filtering after retrieval

    query_embedding = embed_query(query)

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k * 2, 20),  # fetch more, then post-filter
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        # A combined filter ($and of crop+topic) can fail on some ChromaDB
        # versions/collection states. Degrade gracefully — try crop alone
        # (the most important constraint) before ever falling back to a
        # fully unfiltered search. A silent unfiltered fallback previously
        # sat here, which meant a "crop-filtered" query could silently
        # search the ENTIRE corpus with zero warning beyond a log line —
        # this is exactly what let Potato/Apple content leak into Tomato
        # results despite the calling code explicitly requesting a filter.
        logger.warning(f"Filtered search failed ({e}), retrying with crop-only filter")
        crop_only_filter = {"crop": {"$eq": crop}} if crop and crop != "general" else None
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k * 2, 20),
                where=crop_only_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e2:
            logger.error(
                f"Crop-only filtered search ALSO failed ({e2}) — "
                f"returning NO results rather than silently searching the "
                f"full unfiltered corpus. Investigate ChromaDB where-clause "
                f"compatibility."
            )
            return []

    if not results["ids"] or not results["ids"][0]:
        return []

    # Post-filter for disease if specified
    output = []
    for i, doc_id in enumerate(results["ids"][0]):
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        text = results["documents"][0][i]

        # Disease post-filter — a chunk is relevant if EITHER its metadata
        # tag matches OR the disease phrase appears in its own text.
        # Previously: an empty tag list short-circuited the whole check
        # (falsy `doc_diseases` skipped the condition entirely), so any
        # untagged chunk passed automatically regardless of actual
        # relevance — silently filling the result quota with irrelevant
        # high-raw-score chunks before ever reaching a correctly-tagged
        # one further down the ranking.
        if disease:
            doc_diseases = json.loads(meta.get("diseases", "[]"))
            disease_lower = disease.lower()
            matches_tag = any(disease_lower in d for d in doc_diseases)
            matches_text = disease_lower in text.lower()
            if not (matches_tag or matches_text):
                continue

        output.append({
            "id": doc_id,
            "text": text,
            "score": round(1 - distance, 4),  # cosine similarity
            "metadata": meta,
        })

        if len(output) >= top_k:
            break

    return output
