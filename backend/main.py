"""
AgriN — FastAPI backend v4 (RAG + async)
  Stream 1: Plant disease detection (EfficientNet-B3) + RAG grounding
  Stream 2: Crop recommendation (XGBoost + SHAP + regional yield reranking) + RAG grounding
  Advisory: Gemini Flash generates structured, localized advice grounded in NIPHM/ICAR data
  Safety: CIBRC banned pesticide post-validation

Changes from v3:
  - RAG integration: ChromaDB vector store with NIPHM IPM packages
  - CIBRC safety layer: banned pesticide detection in advisory output
  - Async parallelization: climate normals fetch 10 years in parallel (not sequential)
  - Removed dead SoilGrids call (API is officially paused, was wasting 1-2s)
  - Gemini prompts now include RAG context + banned pesticide list
"""

from dotenv import load_dotenv
load_dotenv()

import os
import io
import re
import json
import uuid
import logging
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b3
from PIL import Image
import xgboost as xgb
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("shap not available (import failed) — SHAP explainability disabled, crop predictions still work")
import joblib
import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from google.cloud import firestore, bigquery, storage as gcs
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False
    logging.warning("Google Cloud SDKs not installed — local-only mode")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logging.warning("google-generativeai not installed — Gemini disabled")

# ── RAG imports ──────────────────────────────────────────────────────────
RAG_AVAILABLE = False
try:
    from rag.store import get_collection
    from rag.retrieval import (
        retrieve_for_crop_advisory,
        retrieve_for_disease_advisory,
        format_rag_context,
    )
    from rag.safety import check_advisory, get_banned_list_for_prompt
    RAG_AVAILABLE = True
except ImportError:
    logging.warning("RAG module not available — running without retrieval grounding")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_DIR = os.environ.get("MODEL_DIR", "models")
GCP_PROJECT = os.environ.get("GCP_PROJECT", "")
GCS_BUCKET = os.environ.get("GCS_BUCKET", "agrin-images")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
GEE_SERVICE_ACCOUNT = os.environ.get("GEE_SERVICE_ACCOUNT", "")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEASON_MONTHS = {
    "Kharif": (6, 10),
    "Rabi": (11, 3),
    "Summer": (3, 5),
    "Whole Year": (1, 12),
}

# State-level Soil Health Card averages (primary soil data source for India)
STATE_SOIL_DEFAULTS = {
    "Tamil Nadu":       {"N": 45, "P": 22, "K": 38, "ph": 7.2},
    "Kerala":           {"N": 55, "P": 30, "K": 45, "ph": 5.5},
    "Karnataka":        {"N": 42, "P": 25, "K": 35, "ph": 6.8},
    "Andhra Pradesh":   {"N": 38, "P": 20, "K": 32, "ph": 7.5},
    "Telangana":        {"N": 40, "P": 18, "K": 30, "ph": 7.3},
    "Maharashtra":      {"N": 35, "P": 15, "K": 28, "ph": 7.8},
    "Gujarat":          {"N": 30, "P": 18, "K": 25, "ph": 7.9},
    "Rajasthan":        {"N": 25, "P": 12, "K": 20, "ph": 8.2},
    "Madhya Pradesh":   {"N": 35, "P": 15, "K": 30, "ph": 7.5},
    "Uttar Pradesh":    {"N": 45, "P": 20, "K": 35, "ph": 7.8},
    "Bihar":            {"N": 40, "P": 18, "K": 32, "ph": 7.2},
    "West Bengal":      {"N": 50, "P": 25, "K": 40, "ph": 6.5},
    "Odisha":           {"N": 42, "P": 20, "K": 35, "ph": 6.2},
    "Punjab":           {"N": 50, "P": 25, "K": 40, "ph": 8.0},
    "Haryana":          {"N": 40, "P": 20, "K": 30, "ph": 8.1},
    "Assam":            {"N": 55, "P": 28, "K": 42, "ph": 5.2},
    "Jharkhand":        {"N": 38, "P": 15, "K": 28, "ph": 6.0},
    "Chhattisgarh":     {"N": 35, "P": 12, "K": 25, "ph": 6.5},
    "Goa":              {"N": 48, "P": 22, "K": 38, "ph": 5.8},
    "Himachal Pradesh": {"N": 42, "P": 18, "K": 32, "ph": 6.5},
    "Uttarakhand":      {"N": 40, "P": 16, "K": 30, "ph": 6.8},
    "Jammu And Kashmir":{"N": 38, "P": 15, "K": 28, "ph": 7.0},
}

SUPPORTED_LANGUAGES = {
    "en": "English", "hi": "Hindi", "ta": "Tamil", "te": "Telugu",
    "kn": "Kannada", "ml": "Malayalam", "mr": "Marathi", "bn": "Bengali",
    "gu": "Gujarati", "pa": "Punjabi", "or": "Odia",
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agrin")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="AgriN API", version="4.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global model references
# ---------------------------------------------------------------------------
disease_model = disease_label_map = disease_transform = None
crop_model = crop_explainer = crop_label_map = None
crop_name_map = yield_lookup = crop_features = shap_layout = None
firestore_client = bigquery_client = gcs_client = None


@app.on_event("startup")
async def load_models():
    global disease_model, disease_label_map, disease_transform
    global crop_model, crop_explainer, crop_label_map, crop_name_map
    global yield_lookup, crop_features, shap_layout
    global firestore_client, bigquery_client, gcs_client

    # Disease model
    dw = os.path.join(MODEL_DIR, "best_model.pth")
    dl = os.path.join(MODEL_DIR, "label_map.json")
    if os.path.exists(dw):
        with open(dl) as f:
            disease_label_map = json.load(f)
        num_classes = len(disease_label_map)
        disease_model = efficientnet_b3(weights=None)
        disease_model.classifier = nn.Sequential(
            nn.BatchNorm1d(1536, eps=0.001, momentum=0.01),
            nn.Linear(1536, 256), nn.ReLU(),
            nn.Dropout(p=0.45), nn.Linear(256, num_classes),
        )
        disease_model.load_state_dict(torch.load(dw, map_location=DEVICE, weights_only=True))
        disease_model.to(DEVICE).eval()
        disease_transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        logger.info(f"Disease model loaded: {num_classes} classes, device={DEVICE}")

    # Crop model
    cm = os.path.join(MODEL_DIR, "crop_xgb_model.json")
    if os.path.exists(cm):
        crop_model = xgb.XGBClassifier()
        crop_model.load_model(cm)
        with open(os.path.join(MODEL_DIR, "crop_label_map.json")) as f:
            crop_label_map = {int(k): v for k, v in json.load(f).items()}
        with open(os.path.join(MODEL_DIR, "crop_name_map.json")) as f:
            crop_name_map = json.load(f)
        with open(os.path.join(MODEL_DIR, "yield_lookup.json")) as f:
            yield_lookup = json.load(f)
        with open(os.path.join(MODEL_DIR, "feature_config.json")) as f:
            config = json.load(f)
            crop_features = config["features"]
            shap_layout = config.get("shap_layout", "new")
        ep = os.path.join(MODEL_DIR, "shap_explainer.joblib")
        if SHAP_AVAILABLE and os.path.exists(ep):
            try:
                crop_explainer = joblib.load(ep)
                logger.info("SHAP explainer loaded")
            except Exception as e:
                logger.warning(f"SHAP explainer failed to load: {e} — continuing without it")
                crop_explainer = None
        elif not SHAP_AVAILABLE:
            logger.warning("Skipping SHAP explainer load — shap module unavailable")
        logger.info(f"Crop model loaded: {len(crop_label_map)} classes, {len(crop_features)} features")

    # GCP
    if GCP_AVAILABLE and GCP_PROJECT:
        try:
            firestore_client = firestore.Client(project=GCP_PROJECT)
            bigquery_client = bigquery.Client(project=GCP_PROJECT)
            gcs_client = gcs.Client(project=GCP_PROJECT)
            logger.info("GCP clients initialized")
        except Exception as e:
            logger.warning(f"GCP init failed: {e}")

    if GEMINI_AVAILABLE and GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        logger.info("Gemini configured")

    # RAG
    if RAG_AVAILABLE:
        try:
            col = get_collection()
            logger.info(f"RAG index loaded: {col.count()} chunks")
        except Exception as e:
            logger.warning(f"RAG index not ready: {e}")


# ---------------------------------------------------------------------------
# Reverse geocoding + season detection
# ---------------------------------------------------------------------------
async def reverse_geocode(lat: float, lon: float) -> dict:
    """Reverse geocode lat/lon to Indian state via Nominatim."""
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {"lat": lat, "lon": lon, "format": "json", "zoom": 5, "addressdetails": 1}
    headers = {"User-Agent": "AgriN/4.0 (agricultural-advisory-prototype)"}

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        address = data.get("address", {})
        state = (
            address.get("state") or address.get("province")
            or address.get("region") or address.get("state_district") or ""
        ).strip()
        district = (
            address.get("state_district") or address.get("county")
            or address.get("city") or ""
        ).strip()
        country = address.get("country", "").strip()

        state_fixes = {
            "Tamilnadu": "Tamil Nadu", "Tamil Nādu": "Tamil Nadu",
            "Andhra pradesh": "Andhra Pradesh",
            "Madhya pradesh": "Madhya Pradesh",
            "Uttar pradesh": "Uttar Pradesh",
            "West bengal": "West Bengal",
            "Himachal pradesh": "Himachal Pradesh",
            "Arunachal pradesh": "Arunachal Pradesh",
            "Jammu and Kashmir": "Jammu And Kashmir",
        }
        state = state_fixes.get(state, state)
        if state and state not in STATE_SOIL_DEFAULTS:
            title_state = state.title()
            state = state_fixes.get(title_state, title_state)

        logger.info(f"Geocoded: ({lat}, {lon}) → state={state}, district={district}")
        return {"state": state if state else None, "district": district, "country": country, "source": "nominatim"}
    except Exception as e:
        logger.error(f"Reverse geocoding error: {e}")
        return {"state": None, "district": None, "error": str(e)}


def detect_season(month: int = None) -> str:
    if month is None:
        month = datetime.now().month
    if 6 <= month <= 10:
        return "Kharif"
    elif month >= 11 or month <= 2:
        return "Rabi"
    else:
        return "Summer"


# ---------------------------------------------------------------------------
# Soil data — state averages (SoilGrids removed — API officially paused)
# ---------------------------------------------------------------------------
async def fetch_soil_data(lat: float, lon: float, state: str = None) -> dict:
    """
    Soil data from state-level SHC averages.
    SoilGrids REST API call removed — it's officially paused and was wasting 1-2s.
    Future: nearest-neighbor lookup from geo-tagged SHC data points.
    """
    if state and state in STATE_SOIL_DEFAULTS:
        defaults = STATE_SOIL_DEFAULTS[state]
        return {
            **defaults,
            "source": "state_average",
            "source_detail": f"Soil Health Card state average for {state}",
            "confidence": "moderate",
            "confidence_note": f"Using average soil values for {state}. Enter your Soil Health Card values for better accuracy.",
        }

    logger.warning("Using national average soil defaults")
    return {
        "N": 40, "P": 20, "K": 30, "ph": 6.8,
        "source": "national_average",
        "source_detail": "National average — no state-specific data available",
        "confidence": "low",
        "confidence_note": "Using national average soil values. Please enter your Soil Health Card values for accurate recommendations.",
    }


# ---------------------------------------------------------------------------
# Climate + Weather + NDVI — ASYNC PARALLELIZED
# ---------------------------------------------------------------------------
def _get_season_date_ranges(season: str, years_back: int = 10):
    current_year = datetime.now().year
    start_month, end_month = SEASON_MONTHS.get(season, (1, 12))
    ranges = []
    for y in range(current_year - years_back, current_year):
        if start_month <= end_month:
            s = f"{y}-{start_month:02d}-01"
            e = f"{y}-12-31" if end_month == 12 else f"{y}-{end_month+1:02d}-01"
        else:
            s = f"{y}-{start_month:02d}-01"
            e = f"{y+1}-12-31" if end_month + 1 > 12 else f"{y+1}-{end_month+1:02d}-01"
        ranges.append((s, e, y))
    return ranges


async def _fetch_single_year_climate(client: httpx.AsyncClient,
                                      lat: float, lon: float,
                                      start_date: str, end_date: str,
                                      year: int) -> Optional[dict]:
    """Fetch climate data for a single year. Used in parallel."""
    try:
        resp = await client.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": lat, "longitude": lon,
                "start_date": start_date, "end_date": end_date,
                "daily": "temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum",
                "timezone": "auto",
            }
        )
        if resp.status_code != 200:
            return None
        daily = resp.json().get("daily", {})
        temps = [t for t in (daily.get("temperature_2m_mean") or []) if t is not None]
        humids = [h for h in (daily.get("relative_humidity_2m_mean") or []) if h is not None]
        precips = [p for p in (daily.get("precipitation_sum") or []) if p is not None]
        if temps and humids and precips:
            return {
                "year": year, "temperature": np.mean(temps),
                "humidity": np.mean(humids), "rainfall": sum(precips),
            }
    except Exception as e:
        logger.warning(f"Climate fetch failed for year {year}: {e}")
    return None


async def fetch_climate_normals(lat: float, lon: float, season: str) -> dict:
    """
    Recency-weighted 10-year seasonal climate averages.
    v4: All 10 years fetched in PARALLEL via asyncio.gather (was sequential).
    """
    decay = 0.85
    current_year = datetime.now().year
    date_ranges = _get_season_date_ranges(season, years_back=10)

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            # ── PARALLEL fetch all 10 years at once ──
            tasks = [
                _fetch_single_year_climate(client, lat, lon, sd, ed, yr)
                for sd, ed, yr in date_ranges
            ]
            results = await asyncio.gather(*tasks)

        yearly_data = [r for r in results if r is not None]

        if not yearly_data:
            return {"error": "No climate data retrieved", "source": "open-meteo-historical"}

        yearly_data.sort(key=lambda x: x["year"])
        wt, wh, wr, tw = 0, 0, 0, 0
        for yd in yearly_data:
            w = decay ** (current_year - yd["year"])
            wt += yd["temperature"] * w
            wh += yd["humidity"] * w
            wr += yd["rainfall"] * w
            tw += w

        # Trends
        rainfall_trend = temp_trend = 0
        if len(yearly_data) >= 3:
            x = np.array([yd["year"] for yd in yearly_data], dtype=float)
            xm = x.mean()
            d = max(np.sum((x - xm) ** 2), 1e-8)
            rainfalls = np.array([yd["rainfall"] for yd in yearly_data])
            temps = np.array([yd["temperature"] for yd in yearly_data])
            rainfall_trend = float(np.sum((x - xm) * (rainfalls - rainfalls.mean())) / d)
            temp_trend = float(np.sum((x - xm) * (temps - temps.mean())) / d)

        return {
            "temperature": round(wt / tw, 2),
            "humidity": round(wh / tw, 2),
            "rainfall": round(wr / tw, 1),
            "years_analyzed": len(yearly_data),
            "year_range": f"{yearly_data[0]['year']}–{yearly_data[-1]['year']}",
            "rainfall_trend_per_year": round(rainfall_trend, 2),
            "temperature_trend_per_year": round(temp_trend, 3),
            "source": "open-meteo-historical",
        }
    except Exception as e:
        logger.error(f"Climate error: {e}")
        return {"error": str(e), "source": "open-meteo-historical"}


async def fetch_weather_forecast(lat: float, lon: float) -> dict:
    """7-day forecast — Gemini context only."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat, "longitude": lon,
                    "current": "temperature_2m,relative_humidity_2m,rain",
                    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max",
                    "timezone": "auto", "forecast_days": 7,
                }
            )
            resp.raise_for_status()
            data = resp.json()

        current = data.get("current", {})
        daily = data.get("daily", {})
        precip = daily.get("precipitation_sum", [])

        return {
            "current_temperature": current.get("temperature_2m"),
            "current_humidity": current.get("relative_humidity_2m"),
            "rainfall_7d_total": round(sum(p for p in precip if p), 1),
            "daily_precipitation": precip,
            "daily_rain_probability": daily.get("precipitation_probability_max", []),
            "dates": daily.get("time", []),
            "source": "open-meteo-forecast",
        }
    except Exception as e:
        logger.error(f"Forecast error: {e}")
        return {"error": str(e), "source": "open-meteo-forecast"}


async def fetch_ndvi(lat: float, lon: float) -> dict:
    """NDVI via NASA POWER proxy (or GEE if configured)."""
    try:
        if GEE_SERVICE_ACCOUNT:
            return await _fetch_ndvi_gee(lat, lon)

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                "https://power.larc.nasa.gov/api/temporal/monthly/point",
                params={
                    "parameters": "T2M,PRECTOTCORR", "community": "AG",
                    "longitude": lon, "latitude": lat,
                    "start": str(datetime.now().year - 1),
                    "end": str(datetime.now().year), "format": "json",
                }
            )
            resp.raise_for_status()
            data = resp.json()

        props = data.get("properties", {}).get("parameter", {})
        precip_vals = [v for v in props.get("PRECTOTCORR", {}).values() if v and v > -990]
        temp_vals = [v for v in props.get("T2M", {}).values() if v and v > -990]
        avg_p = np.mean(precip_vals[-3:]) if len(precip_vals) >= 3 else 0
        avg_t = np.mean(temp_vals[-3:]) if len(temp_vals) >= 3 else 25
        ndvi_est = min(0.9, max(0.1, (avg_p / 200) * 0.6 + (1 - abs(avg_t - 25) / 30) * 0.4))

        return {
            "ndvi": round(ndvi_est, 3),
            "interpretation": _interpret_ndvi(ndvi_est),
            "source": "nasa-power-proxy",
        }
    except Exception as e:
        logger.error(f"NDVI error: {e}")
        return {"ndvi": None, "source": "failed"}


async def _fetch_ndvi_gee(lat, lon):
    try:
        import ee
        creds = ee.ServiceAccountCredentials(GEE_SERVICE_ACCOUNT, os.environ.get("GEE_KEY_PATH", "gee-sa.json"))
        ee.Initialize(creds)
        point = ee.Geometry.Point(lon, lat)
        end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        start = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
        col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
               .filterBounds(point).filterDate(start, end)
               .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)))
        ndvi_img = col.map(lambda i: i.addBands(i.normalizedDifference(["B8","B4"]).rename("NDVI"))).select("NDVI").median()
        val = ndvi_img.reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=10).getInfo()
        nv = val.get("NDVI")
        return {"ndvi": round(nv, 3) if nv else None, "interpretation": _interpret_ndvi(nv) if nv else "unknown", "source": "gee-sentinel-2"}
    except Exception as e:
        return {"ndvi": None, "error": str(e), "source": "gee-failed"}


def _interpret_ndvi(ndvi):
    if ndvi is None: return "unknown"
    if ndvi < 0.15: return "barren/bare soil"
    if ndvi < 0.3: return "sparse/degraded vegetation"
    if ndvi < 0.5: return "moderate vegetation"
    if ndvi < 0.7: return "healthy vegetation"
    return "very dense/healthy vegetation"


async def call_gemini(prompt: str, max_retries: int = 2) -> str:
    """
    Calls Gemini with automatic backoff on free-tier rate limits (429).
    The free tier allows only 5 req/min per model — easy to hit during
    active testing/development, so we retry once or twice with a short
    wait rather than surfacing a raw quota error to the farmer.
    """
    if not GEMINI_AVAILABLE or not GEMINI_API_KEY:
        return "Gemini advisory unavailable — API key not configured."

    for attempt in range(max_retries + 1):
        try:
            model = genai.GenerativeModel(GEMINI_MODEL)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "quota" in err_str.lower()

            if is_rate_limit and attempt < max_retries:
                # Parse suggested retry_delay if present, else default backoff
                wait_s = 15
                match = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", err_str)
                if match:
                    wait_s = int(match.group(1)) + 2  # small buffer
                logger.warning(f"Gemini rate limited (attempt {attempt+1}/{max_retries+1}), waiting {wait_s}s...")
                await asyncio.sleep(wait_s)
                continue

            logger.error(f"Gemini error: {e}")
            return f"Advisory generation failed: {e}"

    return "Advisory generation failed: rate limit exceeded after retries."


# ---------------------------------------------------------------------------
# GCP helpers
# ---------------------------------------------------------------------------
async def log_to_firestore(collection: str, data: dict):
    if not firestore_client: return
    try:
        data["timestamp"] = datetime.now(timezone.utc).isoformat()
        firestore_client.collection(collection).document(str(uuid.uuid4())).set(data)
    except Exception as e:
        logger.error(f"Firestore error: {e}")


async def upload_image_to_gcs(file_bytes, filename) -> Optional[str]:
    if not gcs_client: return None
    try:
        bucket = gcs_client.bucket(GCS_BUCKET)
        blob_name = f"uploads/{datetime.now(timezone.utc).strftime('%Y/%m/%d')}/{uuid.uuid4()}_{filename}"
        bucket.blob(blob_name).upload_from_string(file_bytes, content_type="image/jpeg")
        return f"gs://{GCS_BUCKET}/{blob_name}"
    except Exception as e:
        logger.error(f"GCS error: {e}")
        return None


async def get_disease_alerts(state: str, days: int = 7) -> list:
    if not firestore_client: return []
    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        docs = firestore_client.collection("disease_logs").where("timestamp", ">=", cutoff).stream()
        counts = {}
        for doc in docs:
            d = doc.to_dict()
            pred = d.get("prediction", {})
            if d.get("state") == state and pred.get("disease"):
                disease = pred["disease"]
                counts[disease] = counts.get(disease, 0) + 1
        return [
            {"disease": dis, "reports": cnt, "state": state, "severity": "high" if cnt >= 10 else "moderate"}
            for dis, cnt in counts.items() if cnt >= 3
        ]
    except Exception as e:
        logger.error(f"Alert query error: {e}")
        return []


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class CropRecommendRequest(BaseModel):
    lat: float
    lon: float
    language: str = "en"
    season: Optional[str] = None
    state: Optional[str] = None
    N: Optional[float] = None
    P: Optional[float] = None
    K: Optional[float] = None
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    ph: Optional[float] = None
    rainfall: Optional[float] = None


class ManualCropRequest(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float
    state: Optional[str] = None
    season: Optional[str] = None
    language: str = "en"


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def predict_disease(image: Image.Image) -> dict:
    if disease_model is None:
        raise HTTPException(503, "Disease model not loaded")
    inp = disease_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(disease_model(inp), dim=1)[0]
    top5 = torch.argsort(probs, descending=True)[:5]
    preds = []
    for idx in top5:
        i = idx.item()
        info = disease_label_map[str(i)]
        preds.append({
            "rank": len(preds) + 1, "crop": info["crop"],
            "status": info["status"], "disease": info["disease"],
            "confidence": round(probs[i].item(), 4),
        })
    return {"top_prediction": preds[0], "all_predictions": preds}


def predict_crop(features, feature_names, state=None, season=None, top_k=5):
    if crop_model is None:
        raise HTTPException(503, "Crop model not loaded")
    X = np.array(features).reshape(1, -1)
    proba = crop_model.predict_proba(X)[0]
    sv = np.array(crop_explainer.shap_values(X)) if crop_explainer else None

    scored = {}
    for idx in range(len(proba)):
        crop = crop_label_map[idx]
        score = float(proba[idx])
        viab, rinfo = 1.0, None
        if state and season and crop in crop_name_map:
            yc = crop_name_map[crop]
            if yc in yield_lookup:
                sd = yield_lookup[yc].get(state, {}).get(season)
                if sd:
                    viab = sd["viability"]
                    rinfo = sd
        scored[idx] = {"crop": crop, "model_score": round(score, 4),
                       "viability": round(viab, 4),
                       "final_score": round(score * viab, 4), "regional_info": rinfo}

    top = sorted(scored.values(), key=lambda x: x["final_score"], reverse=True)[:top_k]
    results = []
    for rank, c in enumerate(top):
        idx = [k for k, v in crop_label_map.items() if v == c["crop"]][0]
        entry = {"rank": rank + 1, "crop": c["crop"],
                 "model_score": c["model_score"], "viability": c["viability"],
                 "final_score": c["final_score"]}
        if sv is not None:
            cs = sv[0, :, idx] if shap_layout == "new" else sv[idx, 0, :]
            sd = {f: round(float(cs[i]), 4) for i, f in enumerate(feature_names)}
            sd = dict(sorted(sd.items(), key=lambda x: abs(x[1]), reverse=True))
            dom = max(sd, key=lambda k: abs(sd[k]))
            pct = abs(sd[dom]) / max(sum(abs(v) for v in sd.values()), 1e-8)
            entry["shap"] = sd
            entry["dominant_factor"] = {"feature": dom, "percentage": round(pct * 100)}
        if c["regional_info"]:
            entry["regional_data"] = {
                "mean_yield": c["regional_info"].get("mean_yield"),
                "years": c["regional_info"].get("n_years"),
            }
        results.append(entry)

    return {
        "input_features": {f: round(float(features[i]), 2) for i, f in enumerate(feature_names)},
        "location": {"state": state, "season": season},
        "predictions": results,
    }


# ---------------------------------------------------------------------------
# Gemini prompts — v4: RAG-augmented + CIBRC safety
# ---------------------------------------------------------------------------
def _lang_instruction(lang):
    name = SUPPORTED_LANGUAGES.get(lang, "English")
    if lang == "en":
        return "Respond in simple English that a farmer can understand."
    return f"""LANGUAGE INSTRUCTION: You MUST respond ENTIRELY in {name}. 
Every single word of your response must be in {name}. 
Do not use any English words except for scientific/technical terms that have no {name} equivalent.
Use simple, everyday {name} that a rural farmer would understand."""


def build_disease_prompt(prediction, ndvi_data=None, alerts=None,
                         language="en", rag_context=""):
    top = prediction["top_prediction"]
    ndvi = ""
    if ndvi_data and ndvi_data.get("ndvi"):
        ndvi = f"\nVegetation health at location: NDVI {ndvi_data['ndvi']} ({ndvi_data.get('interpretation', '')})"

    alert_text = ""
    if alerts:
        alert_text = "\nREGIONAL ALERTS:\n" + "\n".join(f"- {a['disease']}: {a['reports']} reports recently" for a in alerts)

    # RAG grounding section
    rag_section = ""
    if rag_context:
        banned = get_banned_list_for_prompt() if RAG_AVAILABLE else ""
        rag_section = f"""

OFFICIAL REFERENCE MATERIAL (from NIPHM/ICAR publications — use these for specific recommendations):
---
{rag_context}
---

IMPORTANT: Base your IMMEDIATE ACTION and PREVENTION advice on the reference material above when available.
Cite specific chemical names, dosages, and application methods from the references.
Do NOT recommend any of these BANNED pesticides: {banned}
If the reference material doesn't cover this specific situation, say so and give general advice."""

    return f"""You are an agricultural advisor for Indian farmers.

A farmer's {top['crop']} plant was diagnosed:
- Disease: {top['disease'] or 'Healthy'} (Confidence: {top['confidence']*100:.0f}%)
{ndvi}{alert_text}{rag_section}

Give your response in this EXACT structure with these headers:

**DIAGNOSIS**
What this disease is and how it affects the crop (2-3 sentences)

**IMMEDIATE ACTION**
Numbered steps the farmer should take right now (3-4 steps with specific chemicals and dosages)

**PREVENTION**
How to prevent this in future seasons (3-4 points)

**WHEN TO GET HELP**
When to visit the local Krishi Vigyan Kendra (1-2 sentences)

{_lang_instruction(language)}
Keep it under 300 words. Be specific — name exact products, dosages, timings."""


def build_crop_prompt(context, climate, forecast, soil_source, soil_confidence,
                      ndvi_data=None, language="en", rag_context=""):
    ndvi = ""
    if ndvi_data and ndvi_data.get("ndvi"):
        ndvi = f"\nVegetation: NDVI {ndvi_data['ndvi']} ({ndvi_data.get('interpretation', '')})"

    trend_alert = ""
    rt = climate.get("rainfall_trend_per_year", 0)
    if rt and abs(rt) > 2:
        trend_alert = f"\n⚠ Rainfall {'declining' if rt < 0 else 'increasing'} by {abs(rt):.1f}mm/year over past decade."

    forecast_summary = "Unavailable"
    if forecast.get("dates"):
        total = forecast.get("rainfall_7d_total", 0)
        probs = forecast.get("daily_rain_probability", [])
        max_prob = max(probs) if probs else 0
        forecast_summary = f"{total}mm expected over 7 days, up to {max_prob}% rain probability"

    # RAG grounding section
    rag_section = ""
    if rag_context:
        banned = get_banned_list_for_prompt() if RAG_AVAILABLE else ""
        rag_section = f"""

OFFICIAL REFERENCE MATERIAL (from NIPHM/ICAR/SAU publications):
---
{rag_context}
---

IMPORTANT: Use the reference material above for specific fertilizer dosages, variety recommendations, 
and pest management advice. Cite specific quantities and product names from the references.
Do NOT recommend any of these BANNED pesticides: {banned}"""

    return f"""You are an agricultural advisor for Indian farmers.

SOIL ({soil_source}, confidence: {soil_confidence}):
N={context['input_features'].get('N')}, P={context['input_features'].get('P')}, K={context['input_features'].get('K')}, pH={context['input_features'].get('ph')}

CLIMATE (10-year weighted average for {context['location']['season']}):
Temperature: {climate.get('temperature', '?')}°C | Humidity: {climate.get('humidity', '?')}% | Rainfall: {climate.get('rainfall', '?')}mm
Rainfall trend: {climate.get('rainfall_trend_per_year', '?')}mm/year | Temp trend: {climate.get('temperature_trend_per_year', '?')}°C/year
{trend_alert}

7-DAY FORECAST: {forecast_summary}
{ndvi}

MODEL PREDICTIONS (with SHAP — which factors drove each recommendation):
{json.dumps(context['predictions'], indent=2)}
{rag_section}

Give your response in this EXACT structure with these headers:

**TOP RECOMMENDATIONS**
For each of the top 3 crops: name, why it suits this location, and any risks based on which factor dominates (SHAP). If a crop depends heavily on rainfall and rainfall is declining, flag it.

**RISK ASSESSMENT**
Based on SHAP dominant factors + climate trends, what are the key risks? Suggest drought-tolerant or resilient alternatives if needed.

**FERTILIZER PLAN**
Based on current N/P/K values, specific fertilizer types and quantities per hectare.

**CROP CALENDAR**
For the #1 recommended crop, give a month-by-month timeline:
- Land preparation → Sowing → Key growth stages → Fertilizer schedule → Harvest → Post-harvest

**THIS WEEK**
Based on the 7-day forecast, what should the farmer do or avoid right now?

{"NOTE: Soil data confidence is " + soil_confidence + ". If low, tell the farmer to enter Soil Health Card values for better accuracy." if soil_confidence in ("low", "very low") else ""}

{_lang_instruction(language)}
Keep it under 600 words. Be specific with quantities, dates, product names."""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    rag_status = False
    rag_chunks = 0
    if RAG_AVAILABLE:
        try:
            col = get_collection()
            rag_chunks = col.count()
            rag_status = rag_chunks > 0
        except:
            pass

    return {
        "status": "ok",
        "version": "4.0.0",
        "models": {"disease": disease_model is not None, "crop": crop_model is not None},
        "gemini": GEMINI_AVAILABLE and bool(GEMINI_API_KEY),
        "rag": {"available": rag_status, "chunks": rag_chunks},
        "languages": list(SUPPORTED_LANGUAGES.keys()),
    }


@app.post("/disease")
async def disease_endpoint(
    file: UploadFile = File(...),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
    language: str = Form("en"),
):
    """Stream 1: Crop disease diagnosis from leaf photo + RAG-grounded advisory."""
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file")

    has_loc = lat is not None and lon is not None
    tasks = [upload_image_to_gcs(contents, file.filename or "photo.jpg")]
    tasks.append(fetch_ndvi(lat, lon) if has_loc else asyncio.sleep(0))
    tasks.append(reverse_geocode(lat, lon) if has_loc else asyncio.sleep(0))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    image_url = results[0] if isinstance(results[0], str) else None
    ndvi_data = results[1] if isinstance(results[1], dict) else None
    geo = results[2] if isinstance(results[2], dict) else {}
    state = geo.get("state") if isinstance(geo, dict) else None

    alerts = await get_disease_alerts(state) if state else []
    prediction = predict_disease(image)

    # ── RAG retrieval ──
    rag_context = ""
    if RAG_AVAILABLE:
        top = prediction["top_prediction"]
        try:
            rag_results = retrieve_for_disease_advisory(
                crop=top["crop"],
                disease=top["disease"] or "healthy",
            )
            rag_context = format_rag_context(rag_results)
            logger.info(f"RAG: {len(rag_results)} chunks retrieved for {top['crop']}/{top['disease']}")
        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}")

    prompt = build_disease_prompt(prediction, ndvi_data, alerts, language, rag_context)
    advisory = await call_gemini(prompt)

    # ── CIBRC safety check ──
    if RAG_AVAILABLE:
        safety_result = check_advisory(advisory)
        advisory = safety_result["advisory"]
        if not safety_result["safe"]:
            logger.warning(f"CIBRC: banned pesticides in advisory: {safety_result['banned_found']}")

    top = prediction["top_prediction"]
    return {
        "summary": {
            "crop": top["crop"],
            "disease": top["disease"] or "Healthy",
            "confidence": f"{top['confidence']*100:.0f}%",
            "status": top["status"],
        },
        "advisory": advisory,
        "details": {
            "all_predictions": prediction["all_predictions"],
            "satellite": ndvi_data,
            "location": {"state": state, "district": geo.get("district")} if state else None,
            "alerts": alerts if alerts else None,
            "rag_grounded": bool(rag_context),
        },
        "language": SUPPORTED_LANGUAGES.get(language, "English"),
    }


@app.post("/recommend")
async def recommend_endpoint(req: CropRecommendRequest):
    """
    Stream 2: Crop recommendation from location only.
    Auto-infers state, season, soil, climate, satellite data.
    v4: RAG-grounded advisory + parallelized climate fetch.
    """
    # Step 1: Location intelligence
    geo = await reverse_geocode(req.lat, req.lon)
    state = req.state or geo.get("state")
    season = req.season or detect_season()
    district = geo.get("district", "")

    logger.info(f"Location: state={state}, district={district}, season={season}")

    # Step 2: Parallel data fetch (soil is instant now — no SoilGrids call)
    soil_data, climate, forecast, ndvi = await asyncio.gather(
        fetch_soil_data(req.lat, req.lon, state),
        fetch_climate_normals(req.lat, req.lon, season),
        fetch_weather_forecast(req.lat, req.lon),
        fetch_ndvi(req.lat, req.lon),
    )

    # Step 3: Build features
    features = [
        req.N if req.N is not None else soil_data.get("N", 40),
        req.P if req.P is not None else soil_data.get("P", 20),
        req.K if req.K is not None else soil_data.get("K", 30),
        req.temperature if req.temperature is not None else climate.get("temperature", 25),
        req.humidity if req.humidity is not None else climate.get("humidity", 70),
        req.ph if req.ph is not None else soil_data.get("ph", 6.5),
        req.rainfall if req.rainfall is not None else climate.get("rainfall", 100),
    ]

    soil_source = "manual" if req.N is not None else soil_data.get("source", "unknown")
    soil_conf = "high" if req.N is not None else soil_data.get("confidence", "unknown")

    # Step 4: Predict
    context = predict_crop(features, crop_features, state, season, top_k=5)

    # Step 5: RAG retrieval for top crop
    rag_context = ""
    if RAG_AVAILABLE and context["predictions"]:
        top_crop = context["predictions"][0]["crop"]
        try:
            rag_results = retrieve_for_crop_advisory(
                crop=top_crop, state=state, season=season,
            )
            rag_context = format_rag_context(rag_results)
            logger.info(f"RAG: {len(rag_results)} chunks retrieved for {top_crop}")
        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}")

    # Step 6: Gemini advisory (now with RAG context)
    prompt = build_crop_prompt(context, climate, forecast, soil_source,
                               soil_conf, ndvi, req.language, rag_context)
    advisory = await call_gemini(prompt)

    # Step 7: CIBRC safety check
    if RAG_AVAILABLE:
        safety_result = check_advisory(advisory)
        advisory = safety_result["advisory"]

    # Step 8: Clean response
    top_crop = context["predictions"][0] if context["predictions"] else None

    result = {
        "summary": {
            "location": f"{district}, {state}" if district else state,
            "season": season,
            "season_auto": req.season is None,
            "top_crop": top_crop["crop"] if top_crop else None,
            "confidence": f"{top_crop['final_score']*100:.0f}%" if top_crop else None,
            "soil_confidence": soil_conf,
        },
        "advisory": advisory,
        "predictions": context["predictions"],
        "data_sources": {
            "soil": {"source": soil_source, "confidence": soil_conf, "note": soil_data.get("confidence_note")},
            "climate": {
                "temperature": climate.get("temperature"),
                "humidity": climate.get("humidity"),
                "rainfall": climate.get("rainfall"),
                "rainfall_trend": climate.get("rainfall_trend_per_year"),
                "years": climate.get("years_analyzed"),
            },
            "forecast": {
                "rainfall_7d": forecast.get("rainfall_7d_total"),
                "dates": forecast.get("dates"),
            },
            "satellite": {
                "ndvi": ndvi.get("ndvi") if ndvi else None,
                "interpretation": ndvi.get("interpretation") if ndvi else None,
            },
            "rag_grounded": bool(rag_context),
        },
        "language": SUPPORTED_LANGUAGES.get(req.language, "English"),
    }

    await log_to_firestore("recommendation_logs", {
        "lat": req.lat, "lon": req.lon, "state": state, "district": district,
        "season": season, "top_crop": top_crop["crop"] if top_crop else None,
        "features": context["input_features"], "soil_confidence": soil_conf,
        "language": req.language, "rag_grounded": bool(rag_context),
    })

    return result


@app.post("/recommend/manual")
async def recommend_manual_endpoint(req: ManualCropRequest):
    """Stream 2 (manual): Farmer enters Soil Health Card values."""
    season = req.season or detect_season()
    features = [req.N, req.P, req.K, req.temperature, req.humidity, req.ph, req.rainfall]
    context = predict_crop(features, crop_features, req.state, season, top_k=5)

    # RAG for top crop
    rag_context = ""
    if RAG_AVAILABLE and context["predictions"]:
        top_crop = context["predictions"][0]["crop"]
        try:
            rag_results = retrieve_for_crop_advisory(crop=top_crop, state=req.state, season=season)
            rag_context = format_rag_context(rag_results)
        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}")

    climate = {"note": "Manual entry"}
    forecast = {"note": "No location"}
    prompt = build_crop_prompt(context, climate, forecast, "manual (soil health card)",
                               "high", None, req.language, rag_context)
    advisory = await call_gemini(prompt)

    if RAG_AVAILABLE:
        safety_result = check_advisory(advisory)
        advisory = safety_result["advisory"]

    top = context["predictions"][0] if context["predictions"] else None
    return {
        "summary": {
            "state": req.state, "season": season,
            "top_crop": top["crop"] if top else None,
            "soil_confidence": "high",
        },
        "advisory": advisory,
        "predictions": context["predictions"],
        "rag_grounded": bool(rag_context),
        "language": SUPPORTED_LANGUAGES.get(req.language, "English"),
    }


@app.get("/alerts/{state}")
async def alerts_endpoint(state: str, days: int = 7):
    """Disease outbreak alerts for a state."""
    alerts = await get_disease_alerts(state, days)
    return {"state": state, "period_days": days, "alerts": alerts}


@app.get("/languages")
async def languages_endpoint():
    return {"languages": SUPPORTED_LANGUAGES}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
