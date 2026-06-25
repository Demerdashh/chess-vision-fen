# ── Base Image ─────────────────────────────────────────────────────────────────
# 3.11-slim: minimum version required by numpy>=2.0 and modern ultralytics.
FROM python:3.11-slim

# ── System Dependencies ────────────────────────────────────────────────────────
# libgl1       → libGL.so  — OpenCV codec & drawing support
# libglib2.0-0 → libglib   — Required by OpenCV internals
# Without these, `import cv2` throws: "libGL.so.1: cannot open shared object"
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /chess-vision

COPY requirements.txt .

# ── Install CPU-only PyTorch BEFORE requirements.txt ──────────────────────────
#
# ROOT CAUSE OF THE 2 GB DOWNLOAD PROBLEM:
# pip resolves ultralytics → torch (not installed) → fetches default Linux build:
#   torch           532 MB  ──┐
#   nvidia_cublas   423 MB    │  All CUDA drivers — useless inside a container
#   nvidia_cudnn    366 MB    │  that will never see a GPU.
#   nvidia_cusolver 200 MB  ──┘
#
# FIX: Pre-install the CPU wheel using PyTorch's own index.
# When pip later processes ultralytics from requirements.txt, it finds
# torch already installed and skips it entirely. CUDA packages never touched.
#
# CPU inference is 100% sufficient here — the .pt weights are already trained,
# this container only runs predictions via the FastAPI endpoints.
#
# Download size comparison:
#   Default (CUDA) build → ~2.0 GB
#   CPU-only build       → ~250 MB  ✓
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# ── Install Remaining Dependencies ────────────────────────────────────────────
# torch is already satisfied (CPU). pip skips it and installs everything else:
# fastapi, uvicorn, opencv-contrib-python, ultralytics, numpy, python-multipart
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy Application Files ─────────────────────────────────────────────────────
# Copied AFTER all pip installs.
# This is the critical layer-caching pattern:
# If you change a .py file, Docker replays only the COPY steps below —
# not the expensive pip install layers above (those stay cached).
COPY app/       ./app/
COPY models/    ./models/
COPY templates/ ./templates/
COPY main.py    . 
EXPOSE 8000

# app.apmain:app  →  file: app/apmain.py,  FastAPI instance variable: app
CMD ["uvicorn", "app.apmain:app", "--host", "0.0.0.0", "--port", "8000"]