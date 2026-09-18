# AgriN Backend

FastAPI backend serving both AI streams + Gemini advisory.

## Setup (Local — VS Code)

### 1. Download model files from Kaggle

Place these in `backend/models/`:

**From disease notebook:**
- `best_model.pth` — EfficientNet-B3 weights
- `label_map.json` — disease label map (38 classes)

**From crop recommendation notebook:**
- `crop_xgb_model.json` — XGBoost weights
- `crop_label_map.json` — crop label map (22 classes)
- `crop_name_map.json` — crop name → yield dataset mapping
- `yield_lookup.json` — regional yield statistics
- `shap_explainer.joblib` — SHAP TreeExplainer
- `feature_config.json` — feature names + metadata

### 2. Install dependencies

```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env — add your Gemini API key
# GCP_PROJECT can be empty for local testing (cloud logging will be skipped)
```

### 4. Run locally

```bash
uvicorn main:app --reload --port 8080
```

API docs: http://localhost:8080/docs

### 5. Test endpoints

```bash
# Health check
curl http://localhost:8080/health

# Disease detection
curl -X POST http://localhost:8080/disease \
  -F "file=@test_leaf.jpg"

# Crop recommendation (auto — fetches soil/weather from APIs)
curl -X POST http://localhost:8080/recommend \
  -H "Content-Type: application/json" \
  -d '{"lat": 13.08, "lon": 80.27, "season": "Kharif", "state": "Tamil Nadu"}'

# Crop recommendation (manual — farmer enters soil values)
curl -X POST http://localhost:8080/recommend/manual \
  -H "Content-Type: application/json" \
  -d '{"N": 40, "P": 35, "K": 50, "temperature": 29, "humidity": 78, "ph": 6.2, "rainfall": 120, "state": "Tamil Nadu", "season": "Kharif"}'
```

## Deploy to Cloud Run

```bash
# Build and push
gcloud builds submit --tag gcr.io/YOUR_PROJECT/agrin-backend

# Deploy
gcloud run deploy agrin-backend \
  --image gcr.io/YOUR_PROJECT/agrin-backend \
  --platform managed \
  --region asia-south1 \
  --memory 2Gi \
  --set-env-vars "GEMINI_API_KEY=xxx,GCP_PROJECT=xxx,GCS_BUCKET=agrin-images"
```
