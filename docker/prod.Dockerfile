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

# --- Stage 3: Production (The Tiny Image) ---
FROM python_base AS production

ENV ENVIRONMENT=production \
    DEBUG=False \
    HOST=0.0.0.0 \
    PORT=5000 \
    WORKERS=4 \
    LOG_LEVEL=INFO \
    LOG_FORMAT=json

RUN addgroup -S appuser && adduser -S appuser -G appuser -h /app

USER appuser

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY src ./src

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src"

EXPOSE 5000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]

# Set PATH to use virtual environment
ENV PATH="/app/.venv/bin:$PATH"

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health', timeout=10)" || exit 1

EXPOSE 5000

CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "5000", "--workers", "4"]
