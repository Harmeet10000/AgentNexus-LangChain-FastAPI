# syntax=docker/dockerfile:1.4
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

# --- Builder Stage ---
FROM base AS builder

ENV UV_LINK_MODE=copy

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project && \
    test -d .venv || (echo "ERROR: .venv not created"; exit 1)

# --- Development Stage ---
FROM base AS dev

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src"

COPY pyproject.toml uv.lock ./
COPY src ./src
COPY .env.development* ./

EXPOSE 5000

CMD ["uv", "run", "uvicorn", "src.app.main:app", "--reload", "--reload-dir", "src", "--host", "0.0.0.0", "--port", "5000", "--no-access-log"]
