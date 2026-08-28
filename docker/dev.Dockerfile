# syntax=docker/dockerfile:1.4
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1

WORKDIR /app

# --- Builder Stage ---
FROM base AS builder

ENV UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project && \
    test -d .venv || (echo "ERROR: .venv not created"; exit 1)

# --- Development Stage ---
FROM base AS dev

# ponytail: same version args as prod so `docker inspect` + `/` answer the same question in dev.
ARG GIT_SHA=unknown
ARG BUILD_DATE=unknown
ARG APP_VERSION=1.0.0

LABEL org.opencontainers.image.revision=$GIT_SHA \
      org.opencontainers.image.version=$APP_VERSION \
      org.opencontainers.image.created=$BUILD_DATE

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    GIT_SHA=$GIT_SHA \
    BUILD_DATE=$BUILD_DATE \
    APP_VERSION=$APP_VERSION

COPY pyproject.toml uv.lock ./
COPY src ./src
COPY .env.development* ./

EXPOSE 5000

CMD ["uv", "run", "uvicorn", "src.app.main:app", "--reload", "--reload-dir", "src", "--host", "0.0.0.0", "--port", "5000", "--no-access-log"]
