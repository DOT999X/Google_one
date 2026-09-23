# AgriN RAG — Setup Guide

## Quick Start (3 steps)

### 1. Install dependencies
```bash
pip install -r requirements_rag.txt
```
You should already have `google-generativeai` from your existing setup.

### 2. Download PDFs + Build Index
```bash
# Download NIPHM/ICAR/TNAU PDFs (~30 files, takes 2-3 min)
python rag/download_pdfs.py

# Build the ChromaDB vector index (embeds all chunks via text-embedding-004)
# Requires GEMINI_API_KEY in .env
python -m rag.build_index
```

If some PDFs fail to download (government sites can be flaky), download them manually from the URLs printed and place in `rag/data/pdfs/`.

### 3. Run the server
```bash
python main.py
# or
uvicorn main:app --reload --port 8080
```

The server auto-detects the RAG index at startup. Check `/health` — it now shows:
```json
{
  "rag": {"available": true, "chunks": 1234}
}
```

## How It Works

### Pipeline Flow (with RAG)
```
Farmer input
    │
    ├── [Disease] EfficientNet → prediction
    │       ↓
    │   ChromaDB: filter(crop=Tomato, disease=late_blight) → top chunks
    │       ↓
    │   Gemini prompt + RAG context + CIBRC banned list → advisory
    │       ↓
    │   CIBRC post-validation (flag any banned chemicals)
    │
    ├── [Crop] XGBoost + SHAP → predictions
    │       ↓
    │   ChromaDB: filter(crop=Rice, topic=fertilizer) → top chunks
    │       ↓
    │   Gemini prompt + RAG context + CIBRC banned list → advisory
    │       ↓
    │   CIBRC post-validation
```

### What Changed from v3

| Feature | v3 | v4 |
|---------|----|----|
| Gemini grounding | General LLM knowledge | NIPHM/ICAR official recommendations |
| Pesticide safety | None | CIBRC banned list validation |
| Climate fetch | Sequential (10 API calls in loop) | Parallel (asyncio.gather) |
| SoilGrids | Called then failed (1-2s wasted) | Removed entirely |
| `/health` response | models + gemini | + rag status + chunk count |
| Advisory specificity | "apply fungicide" | "apply Mancozeb 75% WP at 2.5g/L" |

### File Structure
```
rag/
├── __init__.py
├── download_pdfs.py    # one-time: fetch PDFs from NIPHM/ICAR/TNAU
├── build_index.py      # one-time: ingest → chunk → embed → store
├── ingest.py           # PDF extraction + structural chunking + metadata
├── embeddings.py       # text-embedding-004 via Google AI Studio
├── store.py            # ChromaDB with metadata-filtered search
├── retrieval.py        # query routing (crop vs disease advisory)
├── safety.py           # CIBRC banned pesticide validation
└── data/
    ├── pdfs/           # downloaded source PDFs
    ├── chunks/         # processed chunks as JSON (debug)
    ├── chromadb/       # vector index (auto-created)
    └── cibrc_banned.json
```

### Rebuilding the Index
```bash
# If you add new PDFs or want to re-embed:
python -m rag.build_index --rebuild
```

### Adding State-Specific Data Later
Put state SAU PDFs in `rag/data/pdfs/` with a prefix matching the source:
- `tnau_*.pdf` → tagged as TNAU_CPG
- `pau_*.pdf` → tagged as PAU_POP
- `icar_*.pdf` → tagged as ICAR_Advisory
- `ipm_*.pdf` → tagged as NIPHM_IPM

Then rebuild: `python -m rag.build_index --rebuild`

## Swapping to Firestore Vector Search (later)
When you configure GCP, replace `store.py` internals:
- ChromaDB → Firestore collection with vector field
- `build_index()` → write to Firestore with embedding field
- `search()` → Firestore `find_nearest()` with metadata filters
- The `retrieval.py` and `safety.py` layers stay unchanged
