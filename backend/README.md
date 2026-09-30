# AgriN - Farmers Guess. We Don't.

AI-powered crop recommendation and plant disease diagnosis for Indian smallholder farmers. Give it a GPS pin (and optionally a leaf photo), get back a structured, localized advisory grounded in real government agronomy documents, in under 20 seconds.

**Team:** Madras Intelligence

---

## What it does

**Crop Recommendation:** auto-infers your state, season, soil defaults, and 10-year climate normals from just latitude/longitude. An XGBoost model (trained on 2,200 samples, 22 crops) scores soil-crop fit, explained via SHAP feature attribution, then re-ranked against real regional yield data where available.

**Disease Detection:** upload a leaf photo, get a diagnosis from an EfficientNet-B3 classifier fine-tuned on 54K+ PlantVillage images (38 crop-disease classes, 99.7% test accuracy). Includes an out-of-distribution check that flags uncertain identifications even when the model's confidence score looks high, since softmax confidence and actual trustworthiness aren't the same thing.

Both flows are grounded via RAG over real NIPHM, TNAU, PAU, and ICAR agronomy documents, checked against CIBRC's banned-pesticide list before the advisory ships, and delivered in the farmer's own language (11 supported).

## Live deployment

**App:** `https://agrin-backend-734148202950.asia-south1.run.app`

Frontend and backend are served from the same Cloud Run service. The interactive API docs are available at the same URL plus `/docs`.

---

## Architecture

```
Farmer (GPS pin, optional photo, language)
        |
        v
   FastAPI backend (Cloud Run)
        |
        |-- Reverse geocoding (Nominatim) -> state, district
        |-- 10 yr climate normals (Open-Meteo, parallel fetch)
        |-- 7 day forecast (Open-Meteo)
        |-- NDVI proxy (NASA POWER)
        |
        |-- [Crop path]  XGBoost + SHAP -> ranked predictions
        |                       |
        |                RAG retrieval (ChromaDB, crop/topic filtered)
        |                       |
        |-- [Disease path]  EfficientNet-B3 -> prediction + confidence
        |                       |
        |                OOD check (nearest-centroid distance)
        |                       |
        |                RAG retrieval (ChromaDB, crop/disease filtered)
        |                       |
        v
   Gemini Flash (direct HTTP) -> structured, localized advisory
        |
        v
   CIBRC safety check (banned pesticide scan)
        |
        v
   Response -> Frontend
```

## Tech stack

| Layer | Technology |
|---|---|
| Disease model | EfficientNet-B3 (PyTorch), fine-tuned on PlantVillage |
| Crop model | XGBoost + SHAP TreeExplainer |
| OOD detection | Feature-space centroid distance (1536-dim embeddings) |
| RAG | ChromaDB + sentence-transformers/all-MiniLM-L6-v2 (local, no API) |
| LLM | Gemini Flash, called via direct HTTP |
| Backend | FastAPI, Python 3.11 |
| Frontend | Single-file React (CDN, Babel in-browser), no build step |
| Deployment | Docker, Google Cloud Run |

---

## Platform notes

The system is built on Google Cloud Platform end to end. Cloud Run hosts the backend, Firebase Hosting serves the frontend, and Gemini Flash generates every advisory. Firestore, BigQuery, and Cloud Storage integrations are implemented in the codebase for diagnosis logging, regional analytics, and image storage, and are ready to switch on with a project-level configuration step.

## Known limitations

- The disease model was trained on PlantVillage's clean, studio-lit images. Real-world field photos look different, so the out-of-distribution check is a safety net, not a guarantee. It can occasionally flag a correct diagnosis or miss a subtle misclassification.
- A handful of PlantVillage crops only have a healthy class or only have disease classes, not both, so the model's range on those specific crops is narrower.
- Regional yield data covers 12 of 22 crops; the rest use a transparent, clearly-flagged estimate instead of a fabricated perfect score.
- State-specific agronomic guidance currently covers Tamil Nadu and Punjab in depth, alongside national-level coverage everywhere else.

---

## Local development

### 1. Download model files

From your Kaggle training notebooks, place in `backend/models/`:

**Disease model:**
- `best_model.pth`, `label_map.json`

**Crop recommendation model:**
- `crop_xgb_model.json`, `crop_label_map.json`, `crop_name_map.json`
- `yield_lookup.json`, `shap_explainer.joblib`, `feature_config.json`

**OOD detection:**
- `ood_centroids.json` (see `scripts/compute_ood_centroids.py`)

### 2. Install dependencies

```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt -r requirements_rag.txt
```

### 3. Build the RAG index (one-time)

```bash
python rag/download_pdfs.py
python -m rag.build_index
```
See `RAG_SETUP.md` for details.

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env, add your Gemini API key
```

### 5. Run locally

```bash
python main.py
```

API docs: http://localhost:8080/docs

### 6. Test endpoints

```bash
curl http://localhost:8080/health

curl -X POST http://localhost:8080/disease -F "file=@test_leaf.jpg"

curl -X POST http://localhost:8080/recommend \
  -H "Content-Type: application/json" \
  -d '{"lat": 13.08, "lon": 80.27, "language": "en"}'

curl -X POST http://localhost:8080/recommend/manual \
  -H "Content-Type: application/json" \
  -d '{"N": 40, "P": 35, "K": 50, "temperature": 29, "humidity": 78, "ph": 6.2, "rainfall": 120, "state": "Tamil Nadu", "season": "Kharif"}'
```

---

## Deploy to Cloud Run

Full walkthrough in `GCP_DEPLOY.md`. Summary:

```bash
echo -n "YOUR_GEMINI_API_KEY" | gcloud secrets create gemini-api-key --data-file=-

gcloud run deploy agrin-backend \
  --source . \
  --region asia-south1 \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300 \
  --allow-unauthenticated \
  --set-secrets GEMINI_API_KEY=gemini-api-key:latest \
  --set-env-vars GCP_PROJECT=YOUR_PROJECT_ID,GEMINI_MODEL=gemini-3.6-flash
```

4Gi memory and 2 CPU are the minimum needed for torch, ChromaDB, and sentence-transformers to load reliably.

---

## API endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Model, RAG, and OOD status check |
| `/recommend` | POST | Crop recommendation from lat/lon |
| `/recommend/manual` | POST | Crop recommendation from manual soil values |
| `/disease` | POST | Disease diagnosis from a leaf photo |
| `/alerts/{state}` | GET | Regional disease outbreak alerts |
| `/languages` | GET | Supported language list |
| `/docs` | GET | Interactive Swagger UI |

---

## What's next

- Real Soil Health Card data pipeline
- Live NDVI via Sentinel-2 / Google Earth Engine
- Field pilot with real farmers
- Additional State Agricultural University documents for broader state coverage
- Text-to-speech output for non-literate users
