# AgriN backend — Dockerfile for Google Cloud Run
#
# - Cloud Run injects PORT at runtime (defaults to 8080) — the app must
#   listen on whatever PORT Cloud Run gives it.
# - torch is installed from the CPU-only wheel index explicitly — Cloud Run
#   instances are CPU-only, so the default PyPI wheel's bundled CUDA
#   support is dead weight, roughly doubling image size/build time.
# - Model files and the prebuilt rag/data/chromadb/ index are baked into
#   the image at build time — Cloud Run instances are stateless between
#   deployments.

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt requirements_rag.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements_rag.txt

COPY main.py .
COPY rag/ ./rag/
COPY models/ ./models/

ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
