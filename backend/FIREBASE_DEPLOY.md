# Deploying the AgriN frontend to Firebase Hosting

This is the frontend half of your GCP setup — Firebase Hosting is part of
the same Google Cloud project your backend (Cloud Run) lives in, matches
what your own project docs already named as the target, and is genuinely
free for a static single-file app at this scale.

## Step 0 — Deploy the backend first (if you haven't)

The frontend needs a real backend URL to call. Follow `GCP_DEPLOY.md` first
if you haven't deployed to Cloud Run yet — you'll need that live URL for
Step 3 below.

## Step 1 — Install the Firebase CLI

```powershell
npm install -g firebase-tools
firebase login
```

## Step 2 — Initialize Firebase Hosting in a new folder

Create a clean folder just for the frontend deploy (keeps it separate from
your backend repo):

```powershell
mkdir agrin-frontend-deploy
cd agrin-frontend-deploy
firebase init hosting
```

When prompted:
- **"Use an existing project"** → pick the SAME GCP project you used for
  Cloud Run (Firebase and GCP share the same project ID — this is why
  Firebase counts as "the same Google Cloud project," not a separate service)
- **Public directory:** type `public`
- **Configure as a single-page app:** Yes
- **Set up automatic builds with GitHub:** No (not needed for this)
- It'll ask to overwrite `public/index.html` — say No for now, we'll place our own

## Step 3 — Update the API URL and place the file

Open `agrin-frontend.html`, find this line near the top of the `<script>`:
```javascript
const API_BASE = "http://127.0.0.1:8080";
```
Replace it with your actual Cloud Run URL from Step 0:
```javascript
const API_BASE = "https://agrin-backend-xxxxxxxxxx-el.a.run.app";
```

Then rename and place the file:
```powershell
copy agrin-frontend.html public\index.html
```
(adjust the path if your file is elsewhere)

## Step 4 — Deploy

```powershell
firebase deploy --only hosting
```

Takes under a minute for a single static file. It prints your live URL:
```
Hosting URL: https://YOUR-PROJECT-ID.web.app
```

That's now your public frontend, calling your public Cloud Run backend —
the full app, live, shareable with anyone.

## Step 5 — CORS check (should already be fine)

Your `main.py` has `allow_origins=["*"]` in its CORS middleware, so requests
from your new Firebase URL to your Cloud Run URL should work with no
changes needed. If you ever tighten CORS later for security, you'd need to
explicitly allow your Firebase domain.

## Redeploying after frontend changes

Same as Step 4 — edit `public/index.html`, run `firebase deploy --only hosting`
again. Takes seconds since it's one static file.

## Note on the Gemini quota

Same 20 requests/day cap applies here too, shared across everyone who uses
the live URL — if you're sharing this link with judges/testers, that quota
gets consumed by anyone hitting it, not just you personally. Worth keeping
in mind when deciding when to share the public link.
