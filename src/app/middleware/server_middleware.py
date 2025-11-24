import asyncio
import logging
import time
from collections.abc import Callable
from contextvars import ContextVar

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from nanoid import generate
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# Setup logging
logger = logging.getLogger(__name__)

# Context variable for correlation ID
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")

# Prometheus metrics registry
metrics_registry = CollectorRegistry()

# Metrics
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status_code", "project"],
    registry=metrics_registry,
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path", "status_code", "project"],
    registry=metrics_registry,
)

http_requests_in_progress = Gauge(
    "http_requests_in_progress",
    "HTTP requests in progress",
    ["method", "path", "project"],
    registry=metrics_registry,
)

app_up = Gauge(
    "app_up", "Application up status", ["project"], registry=metrics_registry
)

# Set app up on import
app_up.labels(project="langchain-fastapi").set(1)


# ✅ Using decorator-style middleware for better performance
async def correlation_middleware(request: Request, call_next: Callable) -> Response:
    """Add correlation ID to requests for distributed tracing."""
    # Generate unique correlation ID
    correlation_id = generate(size=21)
    correlation_id_var.set(correlation_id)
    request.state.correlation_id = correlation_id

    logger.info(f"[{correlation_id}] {request.method} {request.url.path} started")

    try:
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = correlation_id

        logger.info(
            f"[{correlation_id}] {request.method} {request.url.path} "
            f"completed with status {response.status_code}"
        )

        return response
    except Exception as e:
        logger.error(
            f"[{correlation_id}] {request.method} {request.url.path} "
            f"failed with error: {e!s}",
            exc_info=True,
        )
        raise


def create_metrics_middleware(project_name: str = "langchain-fastapi"):
    """Factory function to create metrics middleware with project name."""

    async def metrics_middleware(request: Request, call_next: Callable) -> Response:
        """Prometheus metrics middleware for monitoring."""
        # Skip metrics endpoint itself to avoid infinite loop
        if request.url.path == "/metrics":
            return await call_next(request)

        method = request.method
        path = request.url.path

        # Track in-progress requests
        http_requests_in_progress.labels(
            method=method, path=path, project=project_name
        ).inc()

        start_time = time.time()
        status_code = 500  # Default to 500 in case of exception

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception as e:
            logger.error(f"Exception in request {method} {path}: {e!s}", exc_info=True)
            raise
        finally:
            duration = time.time() - start_time

            # Record metrics
            http_requests_total.labels(
                method=method,
                path=path,
                status_code=status_code,
                project=project_name,
            ).inc()

            http_request_duration_seconds.labels(
                method=method,
                path=path,
                status_code=status_code,
                project=project_name,
            ).observe(duration)

            # Decrement in-progress
            http_requests_in_progress.labels(
                method=method, path=path, project=project_name
            ).dec()

    return metrics_middleware


def create_timeout_middleware(timeout_seconds: int = 30):
    """Factory function to create timeout middleware with custom timeout."""

    async def timeout_middleware(request: Request, call_next: Callable) -> Response:
        """Timeout requests after specified duration to prevent hanging."""
        try:
            return await asyncio.wait_for(call_next(request), timeout=timeout_seconds)
        except TimeoutError:
            logger.exception(
                f"Request timeout: {request.method} {request.url.path} "
                f"exceeded {timeout_seconds}s"
            )
            # Return JSON response instead of raising exception
            return JSONResponse(
                status_code=408,
                content={
                    "error": "Request Timeout",
                    "message": f"Request took longer than {timeout_seconds} seconds to process",
                    "path": request.url.path,
                },
            )

    return timeout_middleware


def create_security_headers_middleware(
    enable_csp: bool = False, csp_policy: str = "default-src 'self'"
) -> Callable:
    """Factory to create security headers middleware with configurable CSP."""

    async def security_headers_middleware(
        request: Request, call_next: Callable
    ) -> Response:
        """Add security headers to all responses."""
        response = await call_next(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
        if enable_csp:
            response.headers["Content-Security-Policy"] = csp_policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=()"
        )

        return response

    return security_headers_middleware


def get_correlation_id() -> str:
    """Get current correlation ID from context."""
    return correlation_id_var.get()


def get_metrics() -> tuple[bytes, str]:
    """Get Prometheus metrics in text format."""
    return generate_latest(metrics_registry), CONTENT_TYPE_LATEST
