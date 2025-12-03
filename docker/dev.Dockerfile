# --- Stage 1: Base Setup (Slim) ---
FROM python:3.12-slim AS python_base

# Python optimizations
ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

# --- Stage 2: Builder ---
FROM python_base AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# --- Stage 3: Dev Environment ---
FROM python_base AS dev

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src"

COPY pyproject.toml uv.lock ./
COPY src ./src
COPY .env.development* ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

EXPOSE 5000

CMD ["uv", "run", "uvicorn", "src.app.main:app", "--reload", "--host", "0.0.0.0", "--port", "5000"]
