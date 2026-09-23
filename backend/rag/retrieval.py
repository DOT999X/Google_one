"""
AgriN RAG — Retrieval Logic
Routes crop recommendation and disease detection queries
to the right chunks with proper filtering.
"""

import logging
from .store import search

logger = logging.getLogger("agrin.rag.retrieval")

# ── XGBoost (Kaggle Crop Recommendation dataset) names → RAG-indexed
# (PlantVillage/NIPHM-filename-derived) crop names ─────────────────────────
# The crop-recommendation model was trained on a different dataset than the
# NIPHM/PlantVillage corpus, so its 22 output labels don't share the same
# naming convention (e.g. "rice" vs "Rice", "maize" vs "Corn (maize)",
# "grapes" vs "Grape"). Without this mapping, retrieval silently returns
# zero chunks because the exact-match crop filter never matches.
# Crops mapped to None have no NIPHM/TNAU PDF in the corpus yet — retrieval
# correctly returns no results for these rather than guessing.
KAGGLE_TO_RAG_CROP = {
    "rice": "Rice",
    "maize": "Corn (maize)",
    "chickpea": "Chickpea",
    "kidneybeans": None,
    "pigeonpeas": "Pigeonpea",
    "mothbeans": None,
    "mungbean": None,
    "blackgram": None,
    "lentil": "Lentil",
    "pomegranate": None,
    "banana": "Banana",
    "mango": "Mango",
    "grapes": "Grape",
    "watermelon": None,
    "muskmelon": None,
    "apple": "Apple",
    "orange": "Orange",
    "papaya": None,
    "coconut": None,
    "cotton": "Cotton",
    "jute": None,
    "coffee": None,
}


def _resolve_crop_name(crop: str) -> str | None:
    """
    Map a crop name from either taxonomy (Kaggle crop-rec or PlantVillage
    disease) to the RAG-indexed crop name, or None if uncovered.
    """
    if not crop:
        return None
    key = crop.strip().lower()
    if key in KAGGLE_TO_RAG_CROP:
        return KAGGLE_TO_RAG_CROP[key]
    # Already PlantVillage-style (disease flow) — try as-is, then title case
    return crop


def retrieve_for_crop_advisory(crop: str, state: str = None,
                                season: str = None, top_k: int = 4) -> list[dict]:
    """
    Retrieve RAG context for crop recommendation advisory.
    
    Searches for:
    1. Fertilizer/nutrient management for this crop
    2. Cultural practices (sowing, irrigation, spacing)
    3. Variety recommendations
    4. General crop management
    """
    rag_crop = _resolve_crop_name(crop)
    if rag_crop is None:
        logger.info(f"No RAG corpus coverage for crop '{crop}' — skipping retrieval")
        return []

    all_results = []

    # Query 1: Fertilizer and nutrient management
    q1 = f"{rag_crop} fertilizer nutrient management NPK dose"
    results = search(q1, crop=rag_crop, topic="fertilizer", top_k=2)
    all_results.extend(results)

    # Query 2: Cultural practices
    q2 = f"{rag_crop} sowing planting irrigation cultural practices"
    results = search(q2, crop=rag_crop, topic="cultural_practice", top_k=1)
    all_results.extend(results)

    # Query 3: Variety and general management
    q3 = f"{rag_crop} recommended varieties cultivation management"
    results = search(q3, crop=rag_crop, top_k=1)
    for r in results:
        if r["id"] not in {x["id"] for x in all_results}:
            all_results.append(r)

    # Deduplicate and sort by score
    seen_ids = set()
    deduped = []
    for r in all_results:
        if r["id"] not in seen_ids:
            seen_ids.add(r["id"])
            deduped.append(r)

    deduped.sort(key=lambda x: x["score"], reverse=True)
    return deduped[:top_k]


def retrieve_for_disease_advisory(crop: str, disease: str,
                                   top_k: int = 3) -> list[dict]:
    """
    Retrieve RAG context for disease detection advisory.
    
    Searches for:
    1. Exact disease management (chemical + cultural control)
    2. General disease/pest management for the crop
    """
    all_results = []

    # Query 1: Specific disease management
    q1 = f"{crop} {disease} management control treatment spray"
    results = search(q1, crop=crop, disease=disease,
                     topic="disease_management", top_k=2)
    all_results.extend(results)

    # If no exact disease match, broaden
    if len(all_results) < 2:
        q2 = f"{crop} disease pest management IPM control"
        results = search(q2, crop=crop, topic="disease_management", top_k=2)
        for r in results:
            if r["id"] not in {x["id"] for x in all_results}:
                all_results.append(r)

    # Query 3: Pest management (diseases often co-occur with pests)
    q3 = f"{crop} {disease} prevention cultural practice"
    results = search(q3, crop=crop, topic="pest_management", top_k=1)
    for r in results:
        if r["id"] not in {x["id"] for x in all_results}:
            all_results.append(r)

    # Deduplicate and sort
    seen_ids = set()
    deduped = []
    for r in all_results:
        if r["id"] not in seen_ids:
            seen_ids.add(r["id"])
            deduped.append(r)

    deduped.sort(key=lambda x: x["score"], reverse=True)
    return deduped[:top_k]


def format_rag_context(results: list[dict]) -> str:
    """Format retrieved chunks into a prompt-ready string."""
    if not results:
        return ""

    sections = []
    for i, r in enumerate(results):
        source = r["metadata"].get("source_file", "unknown")
        crop = r["metadata"].get("crop", "")
        topic = r["metadata"].get("topic", "")
        header = r["metadata"].get("header", "")

        # Truncate long chunks to keep prompt reasonable
        text = r["text"]
        if len(text) > 1500:
            text = text[:1500] + "..."

        sections.append(
            f"[Reference {i+1}: {source} | {crop} | {topic} | {header}]\n{text}"
        )

    return "\n---\n".join(sections)
