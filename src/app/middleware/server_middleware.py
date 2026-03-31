# import asyncio
import time
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from guard import IPInfoManager, SecurityMiddleware
from guard.models import SecurityConfig
from nanoid import generate
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from app.utils import RedisProtocolAdapter, execution_path, logger, request_state

if TYPE_CHECKING:
    from typing import Any

    from app.config import Settings

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

RATE_LIMIT_EXCLUDED_PATH_PREFIXES = ("/api-docs", "/api-redoc")
RATE_LIMIT_EXCLUDED_PATHS = {"/metrics", "/swagger.json"}


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


class RequestStateLoggingMiddleware:
    """Pure ASGI middleware that keeps request context alive through streaming."""

    def __init__(self, app: Callable[[dict, Callable, Callable], Awaitable]) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        """Track request-scoped logging state for the full ASGI response lifecycle."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = self._read_correlation_id(scope) or generate(size=21)
        state = {
            "request_id": request_id,
            "path": scope["path"],
            "method": scope["method"],
            "user_id": None,
            "layer": "middleware",
            "flow": "start",
        }

        req_token = request_state.set(state)
        flow_token = execution_path.set([])
        scope_state = scope.setdefault("state", {})
        scope_state["correlation_id"] = request_id
        scope_state["request_id"] = request_id

        start_time = time.perf_counter()
        response_started = False
        response_finished = False
        status_code = 500

        with logger.contextualize(**state):
            async def send_wrapper(message: dict) -> None:
                nonlocal response_started, response_finished, status_code

                if message["type"] == "http.response.start":
                    response_started = True
                    status_code = message["status"]
                    headers = list(message.get("headers", []))
                    headers.append((b"x-correlation-id", request_id.encode()))
                    message["headers"] = headers

                if (
                    message["type"] == "http.response.body"
                    and not message.get("more_body", False)
                    and not response_finished
                ):
                    response_finished = True
                    duration_ms = round((time.perf_counter() - start_time) * 1000, 1)
                    logger.bind(
                        layer="http_middleware_exit",
                        status_code=status_code,
                        duration_ms=duration_ms,
                    ).info("Request finished")

                await send(message)

            try:
                logger.info("Request started")
                await self.app(scope, receive, send_wrapper)
            except Exception:
                duration_ms = round((time.perf_counter() - start_time) * 1000, 1)
                logger.bind(
                    layer="http_middleware_exit",
                    status_code=status_code if response_started else 500,
                    duration_ms=duration_ms,
                ).exception("Request failed")
                raise
            finally:
                request_state.reset(req_token)
                execution_path.reset(flow_token)

    @staticmethod
    def _read_correlation_id(scope: dict) -> str | None:
        for key, value in scope.get("headers", []):
            if key == b"x-correlation-id":
                return value.decode()
        return None


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



def _is_streaming_response(response: Response) -> bool:
    content_type = response.headers.get("content-type", "")
    return isinstance(response, StreamingResponse) or content_type.startswith("text/event-stream")


async def apply_fastapi_guard_response_modifier(response: Response) -> Response:
    """Adjust guard-managed headers for streaming responses."""
    if not _is_streaming_response(response):
        return response

    response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
    response.headers["X-Accel-Buffering"] = "no"
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Connection"] = "keep-alive"

    for header_name in ("Cross-Origin-Opener-Policy", "Cross-Origin-Embedder-Policy"):
        if header_name in response.headers:
            del response.headers[header_name]

    return response


def get_security_middleware(app: "FastAPI") -> SecurityMiddleware | None:
    """Walk the built middleware stack and return the FastAPI Guard middleware instance."""
    if app.middleware_stack is None:
        app.middleware_stack = app.build_middleware_stack()

    current = app.middleware_stack
    while current is not None:
        if isinstance(current, SecurityMiddleware):
            return current
        current = getattr(current, "app", None)

    return None


async def initialize_fastapi_guard(app: "FastAPI", settings: "Settings") -> None:
    """Attach app-managed resources to FastAPI Guard after lifespan startup."""
    guard_middleware = get_security_middleware(app)
    if guard_middleware is None:
        logger.warning("FastAPI Guard middleware not found during startup")
        return

    if settings.FASTAPI_GUARD_ENABLE_REDIS and hasattr(app.state, "redis"):
        redis_adapter = RedisProtocolAdapter(redis=app.state.redis)
        guard_middleware.redis_handler = redis_adapter
        guard_middleware.rate_limit_handler.redis_handler = redis_adapter
        guard_middleware.handler_initializer.redis_handler = redis_adapter

    await guard_middleware.initialize()


def build_fastapi_guard_config(settings: "Settings") -> SecurityConfig:
    """Create the FastAPI Guard configuration used by the main app."""
    passive_mode = (
        settings.FASTAPI_GUARD_PASSIVE_MODE
        if settings.FASTAPI_GUARD_PASSIVE_MODE is not None
        else settings.ENVIRONMENT.lower() != "production"
    )
    enforce_https = (
        settings.FASTAPI_GUARD_ENFORCE_HTTPS
        if settings.FASTAPI_GUARD_ENFORCE_HTTPS is not None
        else settings.ENVIRONMENT.lower() == "production"
    )
    security_headers: dict[str, Any] = {
        "enabled": True,
        "hsts": {
            "max_age": 31536000,
            "include_subdomains": True,
            "preload": False,
        },
        "csp": None,
        "frame_options": "DENY",
        "content_type_options": "nosniff",
        "xss_protection": "0",
        "referrer_policy": "strict-origin-when-cross-origin",
        "permissions_policy": "accelerometer=(), camera=(), geolocation=(), microphone=(), payment=()",
        "custom": None,
    }
    geo_ip_handler = None
    if settings.FASTAPI_GUARD_IPINFO_TOKEN and (
        settings.FASTAPI_GUARD_BLOCKED_COUNTRIES or settings.FASTAPI_GUARD_WHITELIST_COUNTRIES
    ):
        geo_ip_handler = IPInfoManager(
            token=settings.FASTAPI_GUARD_IPINFO_TOKEN,
            db_path=Path("data/ipinfo/country_asn.mmdb"),
        )

    return SecurityConfig(
        passive_mode=passive_mode,
        trusted_proxies=settings.FASTAPI_GUARD_TRUSTED_PROXIES,
        trusted_proxy_depth=settings.FASTAPI_GUARD_TRUSTED_PROXY_DEPTH,
        trust_x_forwarded_proto=bool(settings.FASTAPI_GUARD_TRUSTED_PROXIES),
        geo_ip_handler=geo_ip_handler,
        enable_redis=settings.FASTAPI_GUARD_ENABLE_REDIS,
        redis_url=settings.REDIS_URL if settings.FASTAPI_GUARD_ENABLE_REDIS else None,
        whitelist=settings.FASTAPI_GUARD_WHITELIST,
        blacklist=settings.FASTAPI_GUARD_BLACKLIST,
        blocked_user_agents=settings.FASTAPI_GUARD_BLOCKED_USER_AGENTS,
        auto_ban_threshold=settings.FASTAPI_GUARD_AUTO_BAN_THRESHOLD,
        auto_ban_duration=settings.FASTAPI_GUARD_AUTO_BAN_DURATION,
        blocked_countries=settings.FASTAPI_GUARD_BLOCKED_COUNTRIES,
        whitelist_countries=settings.FASTAPI_GUARD_WHITELIST_COUNTRIES,
        block_cloud_providers=set(settings.FASTAPI_GUARD_BLOCK_CLOUD_PROVIDERS),
        custom_log_file=str(settings.LOG_DIR / "security.log"),
        log_format="json" if settings.FASTAPI_GUARD_LOG_FORMAT == "json" else "text",
        rate_limit=settings.RATE_LIMIT_REQUESTS,
        rate_limit_window=settings.RATE_LIMIT_PERIOD,
        enable_rate_limiting=settings.RATE_LIMIT_ENABLED,
        enable_penetration_detection=True,
        enable_ip_banning=True,
        enforce_https=enforce_https,
        enable_cors=True,
        cors_allow_origins=settings.CORS_ORIGINS,
        cors_allow_methods=settings.CORS_ALLOW_METHODS,
        cors_allow_headers=settings.CORS_ALLOW_HEADERS,
        cors_allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
        cors_expose_headers=settings.CORS_EXPOSE_HEADERS,
        cors_max_age=settings.CORS_MAX_AGE,
        security_headers=security_headers,
        exclude_paths=[
            "/",
            "/metrics",
            "/swagger.json",
            "/api-docs",
            "/api-redoc",
            "/openapi.json",
            "/favicon.ico",
        ],
        custom_response_modifier=apply_fastapi_guard_response_modifier,
    )


def get_metrics() -> tuple[bytes, str]:
    """Get Prometheus metrics in text format."""
    return generate_latest(metrics_registry), CONTENT_TYPE_LATEST
