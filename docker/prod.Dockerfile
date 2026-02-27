# syntax=docker/dockerfile:1.4
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# --- Builder Stage ---
FROM base AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev && \
    test -d .venv || (echo "ERROR: .venv not created"; exit 1)

# --- Production Stage ---
FROM base AS production

RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

COPY --chown=appuser:appuser src ./src

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    ENVIRONMENT=production

USER appuser

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/api/v1/health', timeout=10).read()" || exit 1

CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "5000", "--workers", "${WORKERS:-4}"]
