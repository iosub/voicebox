# ============================================================
# Voicebox — Local TTS Server with Web UI (GPU)
# Multi-stage build: Frontend → Python deps → Runtime base → Runtime
# ============================================================

# === Stage 1: Build frontend ===
FROM oven/bun:1 AS frontend

WORKDIR /build

# Copy workspace config and frontend source
COPY package.json bun.lock CHANGELOG.md ./
COPY app/ ./app/
COPY web/ ./web/

# Strip workspaces not needed for web build, and fix trailing comma
RUN sed -i '/"tauri"/d; /"landing"/d' package.json && \
    sed -i -z 's/,\n  ]/\n  ]/' package.json
RUN bun install --no-save
# Build frontend (skip tsc — upstream has pre-existing type errors)
RUN cd web && bunx --bun vite build


# === Stage 2: Build Python dependencies ===
FROM python:3.11-slim AS backend-builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency installation
RUN pip install --no-cache-dir --upgrade pip uv

# Create the virtual environment that the runtime stage will reuse
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY backend/requirements.txt .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    -r requirements.txt 
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python --no-deps chatterbox-tts
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python --no-deps hume-tada
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python \
    git+https://github.com/QwenLM/Qwen3-TTS.git
# Install misaki Japanese/Chinese extras separately to preserve pip cache layer
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python "misaki[ja,zh]>=0.9.4"
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python --no-deps turboquant-gpu
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python "bitsandbytes>=0.45" triton


# === Stage 3: Runtime base ===
FROM python:3.11-slim AS runtime-base

# Create non-root user for security
RUN groupadd -r voicebox && \
    useradd -r -g voicebox -m -s /bin/bash voicebox

WORKDIR /app

# Install only runtime system dependencies
# gcc + libc6-dev are needed by triton (bitsandbytes dep) for JIT compilation at import time
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    gcc \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# triton JIT needs libcuda.so for linking — create stub from the NVIDIA driver lib
RUN ln -sf /usr/lib/wsl/drivers/nvddsi.inf_amd64_dd43ca7b66a30b36/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so 2>/dev/null || \
    echo 'int cuInit(unsigned int f){return 0;}' | gcc -shared -o /usr/lib/x86_64-linux-gnu/libcuda.so -x c -


# === Stage 4: Runtime ===
FROM runtime-base AS runtime

# Copy installed Python environment from builder stage
COPY --from=backend-builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy backend application code
COPY --chown=voicebox:voicebox backend/ /app/backend/

# Copy built frontend from frontend stage
COPY --from=frontend --chown=voicebox:voicebox /build/web/dist /app/frontend/

# Create data directories owned by non-root user
RUN mkdir -p /app/data/generations /app/data/profiles /app/data/cache \
    && chown -R voicebox:voicebox /app/data

# Switch to non-root user
USER voicebox

# Expose the API port
EXPOSE 17493

# Health check — auto-restart if the server hangs
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=60s \
    CMD curl -f http://localhost:17493/health || exit 1

# Start the FastAPI server
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "17493"]
