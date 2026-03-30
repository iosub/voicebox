# ============================================================
# Voicebox — Local TTS Server with Web UI
# 3-stage build: Frontend → Python deps → Runtime
# ============================================================

# === Stage 1: Build frontend ===
FROM oven/bun:1 AS frontend

WORKDIR /build

# Copy workspace manifests first so dependency install stays cached
COPY package.json bun.lock ./
COPY app/package.json ./app/package.json
COPY web/package.json ./web/package.json

# Strip workspaces not needed for web build, and fix trailing comma
RUN sed -i '/"tauri"/d; /"landing"/d' package.json && \
    tr -d '\r' < package.json > package.json.tmp && mv package.json.tmp package.json && \
    sed -i -z 's/,\n  ]/\n  ]/' package.json
RUN bun install --no-save

# Copy frontend source after dependency install so source edits do not bust bun cache
COPY CHANGELOG.md ./
COPY app/ ./app/
COPY web/ ./web/

# Build frontend (skip tsc — upstream has pre-existing type errors)
RUN cd web && bunx --bun vite build


# === Stage 2: Shared Python base ===
# Pin digest to prevent silent cache invalidation from upstream image updates.
# To update: docker buildx imagetools inspect python:3.11-slim --format '{{json .Manifest}}'
FROM python:3.11-slim@sha256:9358444059ed78e2975ada2c189f1c1a3144a5dab6f35bff8c981afb38946634 AS python-base

WORKDIR /app

ENV UV_LINK_MODE=copy
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN python -m venv /opt/venv


# === Stage 3: Build Python dependencies ===
FROM python-base AS backend-builder

COPY --from=ghcr.io/astral-sh/uv:0.6.9 /uv /uvx /bin/

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    -r requirements.txt
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


# === Stage 4: Runtime base ===
FROM python-base AS runtime-base

# Create non-root user for security
RUN groupadd -r voicebox && \
    useradd -r -g voicebox -m -s /bin/bash voicebox

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    sox \
    curl \
    && rm -rf /var/lib/apt/lists/*


# === Stage 5: Runtime ===
FROM runtime-base AS runtime


# Copy installed Python environment from builder stage
COPY --from=backend-builder /opt/venv /opt/venv

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
