# ================================ Builder Stage ================================
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install uv
ADD https://astral.sh/uv/install.sh /install.sh
RUN chmod -R 655 /install.sh && /install.sh && rm /install.sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

COPY pyproject.toml ./
COPY src/ src/
RUN uv sync --no-editable --no-dev

# ============================== Production Stage ==============================
FROM python:3.12-slim AS production

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv .venv

# Copy application code
COPY src/ src/

# Copy .env files if they exist (optional for Railway compatibility)
COPY .env.development* ./

# Set ownership
RUN chown -R appuser:appuser /app

USER appuser

# Set PATH to use virtual environment
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 5000

CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "5000", "--reload"]
