# Dockerfile
# ----------
# CPU-only inference container for the Retail Shelf AI project.
#
# Notes
# -----
# • This image is a CPU inference/development image.
# • For Mac GPU inference, run the application natively with MPS.
# • For CUDA deployment, build a CUDA-compatible image separately.
# • The image uses python:3.11-slim to keep the image small.

FROM python:3.11-slim

# ── System dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ──────────────────────────────────────────────────────
COPY requirements.txt ./

# Install CPU-only PyTorch first (smaller image), then the rest
RUN pip install --no-cache-dir \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
        ultralytics>=8.3.0 \
        opencv-python-headless>=4.8.0 \
        Pillow>=10.0.0 \
        numpy>=1.24.0 \
        pandas>=2.0.0 \
        streamlit>=1.35.0 \
        PyYAML>=6.0 \
        tqdm>=4.66.0 \
        matplotlib>=3.7.0

# ── Copy project source ──────────────────────────────────────────────────────
COPY src/     ./src/
COPY configs/ ./configs/

# Create directories expected by the app. Mount a trained checkpoint at runtime.
RUN mkdir -p data outputs models

# ── Environment ──────────────────────────────────────────────────────────────
# Force CPU; MPS is not available inside Docker on macOS
ENV DEVICE=cpu
ENV PYTORCH_ENABLE_MPS_FALLBACK=0

# ── Streamlit configuration ──────────────────────────────────────────────────
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# ── Default command: launch Streamlit dashboard ──────────────────────────────
EXPOSE 8501
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# ── Alternative commands ─────────────────────────────────────────────────────
# Single image inference:
#   docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs \
#     retail-shelf-ai python src/inference.py --image data/shelf.jpg --device cpu
#
# Run tests:
#   docker run --rm retail-shelf-ai pytest tests/ -v
