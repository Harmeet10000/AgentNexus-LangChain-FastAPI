"""Server middleware for correlation ID, metrics, security, timeout, and rate limiting."""

import asyncio
import time
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

from fastapi import Request, Response
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from nanoid import generate
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from redis.asyncio import Redis
from starlette.middleware.base import BaseHTTPMiddleware

from app.utils.httpError import http_error

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


class CorrelationMiddleware(BaseHTTPMiddleware):
    """Add correlation ID to requests."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        correlation_id = generate(size=21)
        correlation_id_var.set(correlation_id)
        request.state.correlation_id = correlation_id

        response = await call_next(request)
        response.headers["X-Correlation-ID"] = correlation_id

        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """Prometheus metrics middleware."""

    def __init__(self, app: Any, project_name: str = "langchain-fastapi") -> None:
        super().__init__(app)
        self.project_name = project_name

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip metrics endpoint itself
        if request.url.path == "/metrics":
            response: Response = await call_next(request)
            return response

        method = request.method
        path = request.url.path

        # Track in-progress requests
        http_requests_in_progress.labels(
            method=method, path=path, project=self.project_name
        ).inc()

        start_time = time.time()

        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            status_code = 500
            raise
        finally:
            duration = time.time() - start_time

            # Record metrics
            http_requests_total.labels(
                method=method,
                path=path,
                status_code=status_code,
                project=self.project_name,
            ).inc()

            http_request_duration_seconds.labels(
                method=method,
                path=path,
                status_code=status_code,
                project=self.project_name,
            ).observe(duration)

            # Decrement in-progress
            http_requests_in_progress.labels(
                method=method, path=path, project=self.project_name
            ).dec()

        return response


def get_correlation_id() -> str:
    """Get current correlation ID."""
    return correlation_id_var.get()


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to responses (Helmet equivalent)."""

    def __init__(self, app: Any) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        csp_directives = [
            "default-src 'self'",
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
            "script-src 'self'",
            "font-src 'self' https://fonts.gstatic.com",
            "img-src 'self' data: https:",
            "connect-src 'self'",
            "frame-src 'none'",
            "object-src 'none'",
            "media-src 'self'",
            "manifest-src 'self'",
        ]
        response.headers["Content-Security-Policy"] = "; ".join(csp_directives)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains; preload"
        )
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=()"
        )

        return response


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Timeout requests after specified duration."""

    def __init__(self, app: Any, timeout: int = 30) -> None:
        super().__init__(app)
        self.timeout = timeout

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await asyncio.wait_for(call_next(request), timeout=self.timeout)
        except TimeoutError:
            raise http_error(
                "Request took too long to process", 408, request, Exception("Timeout")
            )


async def init_rate_limiter(redis_client: Redis) -> None:
    """Initialize rate limiter with Redis."""
    await FastAPILimiter.init(redis_client)


async def rate_limit_handler(
    request: Request, response: Response, pexpire: int
) -> Response:
    """Custom rate limit exceeded handler."""
    expire_minutes = pexpire // 60000
    raise http_error(
        message=f"Too many requests from this IP, please try again in {expire_minutes} minutes!",
        status_code=429,
        request=request,
    )


# Rate limiter dependency (500 requests per 15 minutes)
rate_limiter = RateLimiter(times=500, seconds=900)


def get_metrics() -> tuple[bytes, str]:
    """Get Prometheus metrics."""
    return generate_latest(metrics_registry), CONTENT_TYPE_LATEST
