# syntax=docker/dockerfile:1.4
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1

WORKDIR /app

# --- Builder Stage ---
FROM base AS builder

# hnswlib (via headroom-ai) needs a C++ compiler to build from source.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev && \
    test -d .venv || (echo "ERROR: .venv not created"; exit 1)

# --- Production Stage ---
FROM base AS production

# ponytail: git-sha versioning — build args become OCI labels + runtime ENV so every
# running container can answer "what commit am I?" (inspect + / + /health). Upgrade
# to cosign/SLSA when supply-chain audit matters.
ARG GIT_SHA=unknown
ARG BUILD_DATE=unknown
ARG APP_VERSION=1.0.0

LABEL org.opencontainers.image.title="langchain-fastapi-production" \
      org.opencontainers.image.revision=$GIT_SHA \
      org.opencontainers.image.version=$APP_VERSION \
      org.opencontainers.image.created=$BUILD_DATE \
      org.opencontainers.image.source="https://github.com/Harmeet10000/langchain-fastapi-production"

RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

COPY --chown=appuser:appuser src ./src

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    ENVIRONMENT=production \
    GIT_SHA=$GIT_SHA \
    BUILD_DATE=$BUILD_DATE \
    APP_VERSION=$APP_VERSION

USER appuser

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/api/v1/health', timeout=10).read()" || exit 1

CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "5000", "--workers", "${WORKERS:-4}"]
