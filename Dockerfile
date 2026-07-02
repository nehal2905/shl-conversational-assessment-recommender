# syntax=docker/dockerfile:1
# ---- builder: install deps + build the FAISS/BM25 index into the image ----
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY data/catalog.json ./data/catalog.json

# Build the index at image build time so data/index ships in the image.
# (Downloads the bge-small ONNX model once; requires network during build.)
RUN PYTHONPATH=/install/lib/python3.11/site-packages python scripts/build_index.py

# ---- runtime -------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

COPY --from=builder /install /usr/local
COPY --from=builder /app/data ./data
COPY app ./app
COPY scripts ./scripts

# fastembed caches the model under /root/.cache — copy so runtime is offline.
COPY --from=builder /root/.cache /root/.cache

EXPOSE 8000

# $PORT is provided by Render/HF Spaces; default 8000 locally.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
