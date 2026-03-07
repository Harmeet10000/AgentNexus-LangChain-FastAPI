import asyncio
import time
from collections.abc import Awaitable, Callable
from contextvars import ContextVar

from fastapi import Request, Response
from fastapi.responses import ORJSONResponse, StreamingResponse
from nanoid import generate
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from app.utils import execution_path, logger, request_state

# Context variable for correlation ID (thread-safe)
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")

# Prometheus metrics registry
metrics_registry = CollectorRegistry()

# Metrics
http_requests_total = Counter(
    name="http_requests_total",
    documentation="Total HTTP requests",
    labelnames=["method", "endpoint", "status_code", "project"],
    registry=metrics_registry,
)

http_request_duration_seconds = Histogram(
    name="http_request_duration_seconds",
    documentation="HTTP request duration in seconds",
    labelnames=["method", "endpoint", "status_code", "project"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=metrics_registry,
)

http_requests_in_progress = Gauge(
    name="http_requests_in_progress",
    documentation="HTTP requests in progress",
    labelnames=["method", "endpoint", "project"],
    registry=metrics_registry,
)

app_up = Gauge(
    name="app_up",
    documentation="Application up status",
    labelnames=["project"],
    registry=metrics_registry
)


def _normalize_path(path: str) -> str:
    """
    Normalize path for metrics to avoid high cardinality.
    Converts /users/123 -> /users/{id}
    """
    # Skip normalization for common paths
    if path in {"/", "/health", "/metrics", "/docs", "/redoc", "/openapi.json"}:
        return path

    # Simple normalization: replace UUIDs and numeric IDs
    parts = path.split("/")
    normalized = []
    for part in parts:
        if part.isdigit() or (len(part) == 36 and part.count("-") == 4):  # UUID
            normalized.append("{id}")
        else:
            normalized.append(part)

    return "/".join(normalized)


# 4. Middleware Implementation
async def log_request_state_middleware(request: Request, call_next: Callable) -> Response:
    """Add request state tracking to all logs."""
    request_id = request.headers.get("X-Correlation-ID") or generate(size=21)

    state = {
        "request_id": request_id,
        "path": request.url.path,
        "method": request.method,
        "user_id": None,
        "layer": "middleware",
        "flow": "start",  # Initialize the flow
    }

    req_token = request_state.set(state)
    flow_token = execution_path.set([])  # Ensure fresh breadcrumbs per request

    request.state.correlation_id = request_id
    start_time = time.perf_counter()

    with logger.contextualize(**state):
        try:
            logger.info("Request started")
            response = await call_next(request)
            duration_ms = round((time.perf_counter() - start_time) * 1000, 1)

            exit_state = state.copy()
            exit_state.update(
                {
                    "layer": "http_middleware_exit",
                    "status_code": response.status_code,
                    "duration_ms": duration_ms,
                }
            )

            logger.bind(**exit_state).info("Request finished")
            response.headers["X-Correlation-ID"] = request_id
            return response  # noqa: TRY300

        except Exception:
            logger.exception("Request failed")
            raise

        finally:
            # Clean up both context variables to prevent memory leaks
            request_state.reset(req_token)
            execution_path.reset(flow_token)


class MetricsMiddleware:
    """Pure ASGI middleware for Prometheus metrics."""

    def __init__(
        self,
        app: Callable[[dict, Callable, Callable], Awaitable],
        project_name: str = "langchain-fastapi",
    ):
        self.app = app
        self.project_name = project_name
        # Set app up status on creation
        app_up.labels(project=project_name).set(1)

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        """ASGI interface."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Skip metrics endpoint to avoid infinite loop
        path = scope["path"]
        if path == "/metrics":
            await self.app(scope, receive, send)
            return

        method = scope["method"]
        endpoint = _normalize_path(path)

        # Track in-progress requests
        http_requests_in_progress.labels(
            method=method, endpoint=endpoint, project=self.project_name
        ).inc()

        start_time = time.perf_counter()
        status_code = 500  # Default to 500 in case of exception

        async def send_wrapper(message: dict) -> None:
            """Wrapper to capture status code and add headers."""
            nonlocal status_code

            if message["type"] == "http.response.start":
                status_code = message["status"]

                # Add process time header
                process_time = time.perf_counter() - start_time
                headers = list(message.get("headers", []))
                headers.append((b"x-process-time", f"{process_time:.3f}".encode()))
                message["headers"] = headers

            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration = time.perf_counter() - start_time

            # Record metrics
            http_requests_total.labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code,
                project=self.project_name,
            ).inc()

            http_request_duration_seconds.labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code,
                project=self.project_name,
            ).observe(duration)

            # Decrement in-progress
            http_requests_in_progress.labels(
                method=method, endpoint=endpoint, project=self.project_name
            ).dec()


class TimeoutMiddleware:
    """Pure ASGI middleware for request timeouts."""

    def __init__(
        self,
        app: Callable[[dict, Callable, Callable], Awaitable],
        timeout_seconds: int = 30,
    ):
        self.app = app
        self.timeout_seconds = timeout_seconds

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        """ASGI interface."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        try:
            await asyncio.wait_for(
                self.app(scope, receive, send),
                timeout=self.timeout_seconds,
            )
        except TimeoutError:
            # Get correlation ID from scope if available
            correlation_id = "unknown"
            for key, value in scope.get("headers", []):
                if key == b"x-correlation-id":
                    correlation_id = value.decode()
                    break

            path = scope["path"]
            method = scope["method"]

            logger.error(
                f"[{correlation_id}] Request timeout: {method} {path} "
                f"exceeded {self.timeout_seconds}s"
            )

            # Send timeout response
            response = ORJSONResponse(
                status_code=408,
                content={
                    "error": "Request Timeout",
                    "message": f"Request took longer than {self.timeout_seconds} seconds",
                    "path": path,
                    "correlationId": correlation_id,
                },
            )

            await response(scope, receive, send)


async def create_security_headers_middleware(request: Request, call_next: Callable) -> Response:
    """Add elite-level security headers to all responses."""
    response = await call_next(request)

    # 1. Apply Base Security
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "0"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = (
        "accelerometer=(), camera=(), geolocation=(), microphone=(), payment=()"
    )
    # response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none';"

    # 2. Check if this is a Streaming Response
    is_stream = (
        isinstance(response, StreamingResponse)
        or response.headers.get("content-type") == "text/event-stream"
    )

    if is_stream:
        # Relax CORP so a separate frontend can read the LLM token stream
        response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"

        # Tell Nginx/Proxies NOT to buffer this response (Crucial for real-time tokens)
        response.headers["X-Accel-Buffering"] = "no"

        # Ensure the stream isn't cached
        response.headers["Cache-Control"] = "no-cache"
        response.headers["Connection"] = "keep-alive"
    else:
        # Apply strict isolation for normal JSON/HTML responses
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"

    # 6. API Cache Lockdown (Crucial for endpoints returning sensitive user data)
    # If the endpoint shouldn't be cached, force the browser and proxies to drop it.
    # (You might want to apply this conditionally based on the route, see the Step Ahead block)
    # response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    # response.headers["Pragma"] = "no-cache"

    return response


def get_metrics() -> tuple[bytes, str]:
    """Get Prometheus metrics in text format."""
    return generate_latest(metrics_registry), CONTENT_TYPE_LATEST
