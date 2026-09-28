# Deploying AgriN to Google Cloud Run

## A note on the billing hold you saw

What you likely encountered is a temporary card-verification hold (often
a few hundred to a couple thousand rupees depending on your bank), not a
real charge — it's typically released back within several business days.
New GCP accounts also usually come with free trial credit on top of Cloud
Run's already-generous free tier (2 million requests/month). Your actual
hackathon usage should cost you close to nothing real, but the hold itself
is a genuine, unavoidable step to enable Cloud Run.

## Before you start

Open `requirements.txt` — delete any `torch`/`torchvision` lines if
present. The Dockerfile installs a CPU-only build explicitly.

## Step 1 — Install the gcloud CLI

https://cloud.google.com/sdk/docs/install, then:
```powershell
gcloud init
```

## Step 2 — Create a project and link billing

```powershell
gcloud projects create agrin-hackathon --name="AgriN"
gcloud config set project agrin-hackathon
```
Link billing at https://console.cloud.google.com/billing

## Step 3 — Enable required APIs

```powershell
gcloud services enable run.googleapis.com cloudbuild.googleapis.com secretmanager.googleapis.com firestore.googleapis.com
```

## Step 4 — Store your Gemini key in Secret Manager

```powershell
echo "YOUR_GEMINI_API_KEY" | gcloud secrets create gemini-api-key --data-file=-
```

## Step 5 — Arrange your files

In your backend repo root (alongside `main.py`), add the `Dockerfile` and
`.gcloudignore`. Confirm:
```
main.py
requirements.txt
requirements_rag.txt
rag/            (code + rag/data/chromadb/ + cibrc_banned.json)
models/         (best_model.pth, label_map.json, crop_xgb_model.json,
                 crop_label_map.json, crop_name_map.json, yield_lookup.json,
                 feature_config.json, shap_explainer.joblib, ood_centroids.json)
Dockerfile
.gcloudignore
```

## Step 6 — Deploy

```powershell
gcloud run deploy agrin-backend `
  --source . `
  --region asia-south1 `
  --memory 4Gi `
  --cpu 2 `
  --timeout 120 `
  --allow-unauthenticated `
  --set-secrets GEMINI_API_KEY=gemini-api-key:latest `
  --set-env-vars GCP_PROJECT=agrin-hackathon,GEMINI_MODEL=gemini-3.6-flash
```

`--memory 4Gi --cpu 2` matters — the Cloud Run default (256Mi) will
OOM-crash on startup given torch + ChromaDB + sentence-transformers.

First deploy takes a few minutes (Cloud Build has to build the image).
Prints your live URL at the end:
```
https://agrin-backend-xxxxxxxxxx-el.a.run.app
```

## Step 7 — Verify

```
https://YOUR-SERVICE-URL/health
```
Confirm `models.crop`, `models.disease`, `rag.available`, and
`ood_detection.available` are all `true`. Then `/docs` for Swagger UI.

Optionally enable Firestore for the disease-alerts/logging features
already coded in `main.py`: console.cloud.google.com/firestore → Create
Database → Native mode, same region as Step 6.

## Step 8 — Redeploying

Same command as Step 6 — replaces the running version in place.

## Known limitations

- **Cold starts:** scales to zero by default; first request after idle
  takes 20-40s. Hit `/health` a minute before a live demo to warm it up.
  (`--min-instances 1` avoids this but runs outside the pure free tier.)
- **Gemini's 20/day quota** still applies, shared across anyone hitting
  the live URL.
