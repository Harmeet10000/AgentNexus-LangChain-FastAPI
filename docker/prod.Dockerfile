# Multi-stage Production Dockerfile
# Stage 1: Build dependencies
FROM python:3.12-alpine as builder

# Set build environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_CACHE_DIR=/tmp/uv-cache

# Install uv and build dependencies for Alpine
RUN apk add --no-cache --virtual .build-deps \
    gcc \
    musl-dev \
    libffi-dev \
    openssl-dev \
    python3-dev \
    build-base \
    curl

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Create virtual environment with uv
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install Python dependencies with uv
COPY pyproject.toml requirements.txt* ./
RUN if [ -f "pyproject.toml" ]; then \
    uv pip install -e .; \
    else \
    uv pip install -r requirements.txt; \
    fi

# Remove build dependencies and uv cache to reduce image size
RUN apk del .build-deps && \
    rm -rf /tmp/uv-cache /root/.cargo

# Stage 2: Production Runtime
FROM python:3.12-alpine as runtime

# Install only essential runtime dependencies
RUN apk add --no-cache \
    ca-certificates \
    tzdata \
    && rm -rf /var/cache/apk/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set runtime environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app/src" \
    PORT=5000

# Create non-root user with specific UID/GID for security
RUN addgroup -g 1001 -S appuser && \
    adduser -u 1001 -S appuser -G appuser -h /app -s /bin/false

# Set working directory
WORKDIR /app

# Copy application code with proper ownership
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser pyproject.toml ./

# Create necessary directories with proper permissions
RUN mkdir -p /app/logs /app/tmp && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health', timeout=10)" || exit 1

# Expose port
EXPOSE 5000

# Use exec form for better signal handling
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "5000", "--workers", "1"]
