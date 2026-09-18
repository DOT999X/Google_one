"""
AgriN — FastAPI backend serving both streams:
  Stream 1: Plant disease detection (EfficientNet-B3)
  Stream 2: Crop recommendation (XGBoost + SHAP + regional yield reranking)
  Advisory: Gemini Flash generates localized advice from model outputs

External APIs:
  - SoilGrids (ISRIC) — soil properties from lat/lon
  - Open-Meteo — weather forecast from lat/lon
  - Gemini Flash — natural language advisory
  - Firestore — diagnosis/recommendation logs
  - BigQuery — aggregated analytics
  - Cloud Storage — uploaded crop images
"""

import os
import io
import json
import uuid
import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b3
from PIL import Image
import xgboost as xgb
import shap
import joblib
import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Google Cloud imports — graceful fallback for local dev
try:
    from google.cloud import firestore, bigquery, storage as gcs
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False
    logging.warning("Google Cloud SDKs not installed — running in local-only mode")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logging.warning("google-generativeai not installed — Gemini advisory disabled")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_DIR = os.environ.get("MODEL_DIR", "models")
GCP_PROJECT = os.environ.get("GCP_PROJECT", "")
GCS_BUCKET = os.environ.get("GCS_BUCKET", "agrin-images")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agrin")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="AgriN API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model loading (runs once on startup)
# ---------------------------------------------------------------------------
# -- Disease model --
disease_model = None
disease_label_map = None
disease_transform = None

# -- Crop model --
crop_model = None
crop_explainer = None
crop_label_map = None
crop_name_map = None
yield_lookup = None
crop_features = None
shap_layout = None

# -- GCP clients --
firestore_client = None
bigquery_client = None
gcs_client = None


@app.on_event("startup")
async def load_models():
    global disease_model, disease_label_map, disease_transform
    global crop_model, crop_explainer, crop_label_map, crop_name_map
    global yield_lookup, crop_features, shap_layout
    global firestore_client, bigquery_client, gcs_client

    # ---- Disease model (EfficientNet-B3) ----
    disease_weights_path = os.path.join(MODEL_DIR, "best_model.pth")
    disease_labels_path = os.path.join(MODEL_DIR, "label_map.json")

    if os.path.exists(disease_weights_path):
        with open(disease_labels_path) as f:
            disease_label_map = json.load(f)

        num_classes = len(disease_label_map)
        disease_model = efficientnet_b3(weights=None)
        disease_model.classifier = nn.Sequential(
            nn.BatchNorm1d(1536, eps=0.001, momentum=0.01),
            nn.Linear(1536, 256),
            nn.ReLU(),
            nn.Dropout(p=0.45),
            nn.Linear(256, num_classes),
        )
        disease_model.load_state_dict(
            torch.load(disease_weights_path, map_location=DEVICE, weights_only=True)
        )
        disease_model.to(DEVICE)
        disease_model.eval()

        disease_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        logger.info(f"Disease model loaded: {num_classes} classes, device={DEVICE}")
    else:
        logger.warning(f"Disease model not found at {disease_weights_path} — stream 1 disabled")

    # ---- Crop model (XGBoost + SHAP) ----
    crop_model_path = os.path.join(MODEL_DIR, "crop_xgb_model.json")
    crop_labels_path = os.path.join(MODEL_DIR, "crop_label_map.json")
    crop_namemap_path = os.path.join(MODEL_DIR, "crop_name_map.json")
    yield_lookup_path = os.path.join(MODEL_DIR, "yield_lookup.json")
    explainer_path = os.path.join(MODEL_DIR, "shap_explainer.joblib")
    feature_config_path = os.path.join(MODEL_DIR, "feature_config.json")

    if os.path.exists(crop_model_path):
        crop_model = xgb.XGBClassifier()
        crop_model.load_model(crop_model_path)

        with open(crop_labels_path) as f:
            crop_label_map = {int(k): v for k, v in json.load(f).items()}

        with open(crop_namemap_path) as f:
            crop_name_map = json.load(f)

        with open(yield_lookup_path) as f:
            yield_lookup = json.load(f)

        with open(feature_config_path) as f:
            config = json.load(f)
            crop_features = config["features"]
            shap_layout = config.get("shap_layout", "new")

        if os.path.exists(explainer_path):
            crop_explainer = joblib.load(explainer_path)
            logger.info("SHAP explainer loaded")
        else:
            logger.warning("SHAP explainer not found — SHAP disabled")

        logger.info(f"Crop model loaded: {len(crop_label_map)} classes, {len(crop_features)} features")
    else:
        logger.warning(f"Crop model not found at {crop_model_path} — stream 2 disabled")

    # ---- GCP clients ----
    if GCP_AVAILABLE and GCP_PROJECT:
        try:
            firestore_client = firestore.Client(project=GCP_PROJECT)
            bigquery_client = bigquery.Client(project=GCP_PROJECT)
            gcs_client = gcs.Client(project=GCP_PROJECT)
            logger.info("GCP clients initialized")
        except Exception as e:
            logger.warning(f"GCP init failed: {e} — running without cloud services")

    # ---- Gemini ----
    if GEMINI_AVAILABLE and GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        logger.info("Gemini configured")


# ---------------------------------------------------------------------------
# External API helpers
# ---------------------------------------------------------------------------
async def fetch_soil_data(lat: float, lon: float) -> dict:
    """Query SoilGrids REST API for soil properties at a location."""
    url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
    params = {
        "lat": lat,
        "lon": lon,
        "property": ["nitrogen", "phh2o", "clay", "sand", "silt", "soc", "cec"],
        "depth": "0-5cm",
        "value": "mean",
    }
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        layers = data.get("properties", {}).get("layers", [])
        soil = {}
        for layer in layers:
            name = layer["name"]
            depths = layer.get("depths", [{}])
            if depths:
                values = depths[0].get("values", {})
                soil[name] = values.get("mean")

        # Map SoilGrids outputs to model features (approximate)
        # nitrogen: cg/kg → kg/ha (rough: multiply by ~0.1 for topsoil)
        # phh2o: pH*10 → pH
        nitrogen_val = (soil.get("nitrogen") or 500) / 10.0  # approximate N in kg/ha
        ph_val = (soil.get("phh2o") or 65) / 10.0
        # P and K aren't in SoilGrids — estimate from CEC and clay
        cec_val = soil.get("cec") or 150
        clay_val = soil.get("clay") or 200
        p_estimate = max(5, min(145, cec_val / 10.0 * 3.5))  # rough heuristic
        k_estimate = max(5, min(205, clay_val / 10.0 * 2.5))  # rough heuristic

        return {
            "N": round(nitrogen_val, 1),
            "P": round(p_estimate, 1),
            "K": round(k_estimate, 1),
            "ph": round(ph_val, 2),
            "raw_soilgrids": soil,
            "source": "soilgrids",
        }
    except Exception as e:
        logger.error(f"SoilGrids API error: {e}")
        return {"error": str(e), "source": "soilgrids"}


async def fetch_weather(lat: float, lon: float) -> dict:
    """Query Open-Meteo for current weather + 7-day forecast."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,rain",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "auto",
        "forecast_days": 7,
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        current = data.get("current", {})
        daily = data.get("daily", {})

        # 7-day rainfall total
        precip_daily = daily.get("precipitation_sum", [])
        rainfall_7d = sum(p for p in precip_daily if p is not None)
        rainfall_uncertainty = max(precip_daily) - min(precip_daily) if precip_daily else 0

        return {
            "temperature": current.get("temperature_2m", 25.0),
            "humidity": current.get("relative_humidity_2m", 70.0),
            "rainfall": round(rainfall_7d, 1),
            "rainfall_uncertainty": round(rainfall_uncertainty, 1),
            "forecast_days": len(precip_daily),
            "daily_precip": precip_daily,
            "source": "open-meteo",
        }
    except Exception as e:
        logger.error(f"Open-Meteo API error: {e}")
        return {"error": str(e), "source": "open-meteo"}


async def call_gemini(prompt: str) -> str:
    """Call Gemini Flash for advisory generation."""
    if not GEMINI_AVAILABLE or not GEMINI_API_KEY:
        return "Gemini advisory unavailable — API key not configured."
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return f"Advisory generation failed: {e}"


# ---------------------------------------------------------------------------
# GCP helpers
# ---------------------------------------------------------------------------
async def log_to_firestore(collection: str, data: dict):
    """Log a prediction to Firestore."""
    if not firestore_client:
        return
    try:
        data["timestamp"] = datetime.now(timezone.utc).isoformat()
        firestore_client.collection(collection).document(str(uuid.uuid4())).set(data)
    except Exception as e:
        logger.error(f"Firestore log error: {e}")


async def upload_image_to_gcs(file_bytes: bytes, filename: str) -> Optional[str]:
    """Upload an image to Cloud Storage and return its public URL."""
    if not gcs_client:
        return None
    try:
        bucket = gcs_client.bucket(GCS_BUCKET)
        blob_name = f"uploads/{datetime.now(timezone.utc).strftime('%Y/%m/%d')}/{uuid.uuid4()}_{filename}"
        blob = bucket.blob(blob_name)
        blob.upload_from_string(file_bytes, content_type="image/jpeg")
        return f"gs://{GCS_BUCKET}/{blob_name}"
    except Exception as e:
        logger.error(f"GCS upload error: {e}")
        return None


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class CropRecommendRequest(BaseModel):
    lat: float
    lon: float
    season: str = "Kharif"
    state: Optional[str] = None
    # Optional manual overrides (if farmer has soil test report)
    N: Optional[float] = None
    P: Optional[float] = None
    K: Optional[float] = None
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    ph: Optional[float] = None
    rainfall: Optional[float] = None


class ManualCropRequest(BaseModel):
    """For when farmer enters soil values manually (from soil health card)."""
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float
    state: Optional[str] = None
    season: str = "Kharif"


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def predict_disease(image: Image.Image) -> dict:
    """Run disease detection on a PIL image."""
    if disease_model is None:
        raise HTTPException(503, "Disease model not loaded")

    input_tensor = disease_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = disease_model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]

    top5_indices = torch.argsort(probs, descending=True)[:5]
    predictions = []
    for idx in top5_indices:
        idx_int = idx.item()
        info = disease_label_map[str(idx_int)]
        predictions.append({
            "rank": len(predictions) + 1,
            "class_name": info["class_name"],
            "crop": info["crop"],
            "status": info["status"],
            "disease": info["disease"],
            "confidence": round(probs[idx_int].item(), 4),
        })

    return {
        "top_prediction": predictions[0],
        "all_predictions": predictions,
    }


def predict_crop(features: list, feature_names: list, state: str = None,
                 season: str = None, top_k: int = 5) -> dict:
    """Run crop recommendation with SHAP and regional reranking."""
    if crop_model is None:
        raise HTTPException(503, "Crop model not loaded")

    X_input = np.array(features).reshape(1, -1)

    # Predict
    proba = crop_model.predict_proba(X_input)[0]

    # SHAP
    shap_breakdown_all = {}
    if crop_explainer:
        sample_shap = np.array(crop_explainer.shap_values(X_input))

    # Regional reranking
    reranked = {}
    for idx in range(len(proba)):
        crop_name = crop_label_map[idx]
        base_score = float(proba[idx])

        viability = 1.0
        regional_info = None

        if state and season and crop_name in crop_name_map:
            yield_crop = crop_name_map[crop_name]
            if yield_crop in yield_lookup:
                state_data = yield_lookup[yield_crop].get(state, {})
                season_data = state_data.get(season, None)
                if season_data:
                    viability = season_data["viability"]
                    regional_info = season_data

        reranked[idx] = {
            "crop": crop_name,
            "model_score": round(base_score, 4),
            "regional_viability": round(viability, 4),
            "final_score": round(base_score * viability, 4),
            "regional_info": regional_info,
        }

    sorted_crops = sorted(reranked.values(), key=lambda x: x["final_score"], reverse=True)

    # Build output with SHAP
    predictions = []
    for rank, crop_data in enumerate(sorted_crops[:top_k]):
        idx = [k for k, v in crop_label_map.items() if v == crop_data["crop"]][0]

        entry = {
            "rank": rank + 1,
            "crop": crop_data["crop"],
            "model_score": crop_data["model_score"],
            "regional_viability": crop_data["regional_viability"],
            "final_score": crop_data["final_score"],
        }

        if crop_explainer:
            if shap_layout == "new":
                crop_shap = sample_shap[0, :, idx]
            else:
                crop_shap = sample_shap[idx, 0, :]

            shap_dict = {f: round(float(crop_shap[i]), 4) for i, f in enumerate(feature_names)}
            shap_dict = dict(sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True))

            dominant = max(shap_dict, key=lambda k: abs(shap_dict[k]))
            total_shap = sum(abs(v) for v in shap_dict.values())
            dominant_pct = abs(shap_dict[dominant]) / max(total_shap, 1e-8)

            entry["shap_breakdown"] = shap_dict
            entry["dominant_factor"] = f"{dominant} ({dominant_pct:.0%} of decision)"

        if crop_data["regional_info"]:
            entry["regional_info"] = crop_data["regional_info"]

        predictions.append(entry)

    return {
        "input_features": {f: round(float(features[i]), 2) for i, f in enumerate(feature_names)},
        "location": {"state": state, "season": season},
        "top_predictions": predictions,
    }


# ---------------------------------------------------------------------------
# Gemini prompt builders
# ---------------------------------------------------------------------------
def build_disease_prompt(prediction: dict, image_context: str = "") -> str:
    top = prediction["top_prediction"]
    return f"""You are an agricultural advisor helping Indian farmers identify and treat crop diseases.

A farmer uploaded a photo of their {top['crop']} plant. The AI disease detection model identified:
- Disease: {top['disease'] or 'Healthy'}
- Confidence: {top['confidence']*100:.1f}%
- Status: {top['status']}

Other possibilities:
{json.dumps(prediction['all_predictions'][1:3], indent=2)}

Provide in clear, actionable language:
1. What this disease is and how it affects the crop
2. Immediate treatment steps the farmer should take
3. Preventive measures for the future
4. When to consult an agricultural extension officer

If the plant is healthy, congratulate the farmer and give maintenance tips.
Respond in simple English that a farmer can understand. Keep it under 300 words."""


def build_crop_prompt(context: dict, weather: dict, soil_source: str) -> str:
    return f"""You are an agricultural advisor for Indian farmers. Given the following data, provide:
1. Top 3 crop recommendations with reasoning
2. Risk assessment based on which features dominate each prediction (see SHAP breakdown)
3. Regenerative/sustainable farming advice for the recommended crops
4. Fertilizer recommendations based on current soil nutrient levels

Model predictions with SHAP explanations:
{json.dumps(context, indent=2)}

Weather data (source: Open-Meteo):
{json.dumps(weather, indent=2)}

Soil data source: {soil_source}

IMPORTANT: Check the "dominant_factor" for each crop. If rainfall or humidity dominates and the weather forecast shows high uncertainty, flag this as a risk and suggest drought-tolerant alternatives.

Respond in clear, actionable language a farmer can follow. Keep it under 400 words."""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "disease_model": disease_model is not None,
        "crop_model": crop_model is not None,
        "gemini": GEMINI_AVAILABLE and bool(GEMINI_API_KEY),
        "gcp": GCP_AVAILABLE and bool(GCP_PROJECT),
        "device": str(DEVICE),
    }


@app.post("/disease")
async def disease_endpoint(file: UploadFile = File(...)):
    """
    Stream 1: Upload a crop leaf image → disease prediction + Gemini advisory.
    """
    # Read and validate image
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file")

    # Upload to GCS
    image_url = await upload_image_to_gcs(contents, file.filename or "unknown.jpg")

    # Predict
    prediction = predict_disease(image)

    # Gemini advisory
    prompt = build_disease_prompt(prediction)
    advisory = await call_gemini(prompt)

    result = {
        "prediction": prediction,
        "advisory": advisory,
        "image_url": image_url,
    }

    # Log to Firestore
    await log_to_firestore("disease_logs", {
        "prediction": prediction["top_prediction"],
        "confidence": prediction["top_prediction"]["confidence"],
        "image_url": image_url,
    })

    return result


@app.post("/recommend")
async def recommend_endpoint(req: CropRecommendRequest):
    """
    Stream 2: Location + season → soil/weather APIs → crop recommendation + SHAP + Gemini advisory.
    """
    # Fetch soil and weather data in parallel
    import asyncio
    soil_task = fetch_soil_data(req.lat, req.lon)
    weather_task = fetch_weather(req.lat, req.lon)
    soil_data, weather_data = await asyncio.gather(soil_task, weather_task)

    # Build feature vector — use manual overrides if provided, else API data
    features = [
        req.N if req.N is not None else soil_data.get("N", 50),
        req.P if req.P is not None else soil_data.get("P", 50),
        req.K if req.K is not None else soil_data.get("K", 50),
        req.temperature if req.temperature is not None else weather_data.get("temperature", 25),
        req.humidity if req.humidity is not None else weather_data.get("humidity", 70),
        req.ph if req.ph is not None else soil_data.get("ph", 6.5),
        req.rainfall if req.rainfall is not None else weather_data.get("rainfall", 100),
    ]

    soil_source = "manual" if req.N is not None else soil_data.get("source", "unknown")

    # Determine state from coordinates if not provided
    state = req.state
    # TODO: reverse geocode lat/lon to Indian state if state is None

    # Predict
    context = predict_crop(
        features=features,
        feature_names=crop_features,
        state=state,
        season=req.season,
        top_k=5,
    )

    # Gemini advisory
    prompt = build_crop_prompt(context, weather_data, soil_source)
    advisory = await call_gemini(prompt)

    result = {
        "recommendation": context,
        "weather": weather_data,
        "soil": soil_data,
        "advisory": advisory,
    }

    # Log to Firestore
    await log_to_firestore("recommendation_logs", {
        "lat": req.lat,
        "lon": req.lon,
        "state": state,
        "season": req.season,
        "top_crop": context["top_predictions"][0]["crop"] if context["top_predictions"] else None,
        "features": context["input_features"],
    })

    return result


@app.post("/recommend/manual")
async def recommend_manual_endpoint(req: ManualCropRequest):
    """
    Stream 2 (manual): Farmer enters soil test values directly (from Soil Health Card).
    """
    features = [req.N, req.P, req.K, req.temperature, req.humidity, req.ph, req.rainfall]

    context = predict_crop(
        features=features,
        feature_names=crop_features,
        state=req.state,
        season=req.season,
        top_k=5,
    )

    # Still fetch weather for Gemini context
    weather_data = {"note": "Manual entry — no location-based weather"}

    prompt = build_crop_prompt(context, weather_data, "manual (soil health card)")
    advisory = await call_gemini(prompt)

    result = {
        "recommendation": context,
        "advisory": advisory,
    }

    await log_to_firestore("recommendation_logs", {
        "state": req.state,
        "season": req.season,
        "top_crop": context["top_predictions"][0]["crop"] if context["top_predictions"] else None,
        "features": context["input_features"],
        "source": "manual",
    })

    return result


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
