# ruff: noqa: E402 — dotenv must load before modules that cache Settings

from dotenv import load_dotenv

# Load environment variables before importing modules that construct cached
# settings at import time (models, LangChain callbacks, and routers).
load_dotenv(dotenv_path=".env.development")

from typing import TYPE_CHECKING

from fastapi import FastAPI, Request, status
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import Response
from guard import SecurityMiddleware
from opentelemetry.instrumentation.asgi import OpenTelemetryMiddleware
from returns.result import Failure, Success

from .api import v1_router, v2_router
from .config import get_settings
from .features.health.router import deep_health_router
from .lifecycle import lifespan
from .middleware import (
    ApiDeprecationMiddleware,
    RequestStateLoggingMiddleware,
    build_fastapi_guard_config,
    default_span_details,
    get_metrics,
    register_exception_handlers,
)
from .shared.langchain_layer import configure_langsmith
from .shared.otel import setup_otel
from .shared.result import APIResponse, NotFoundError, render_result
from .utils import logger, setup_logging

if TYPE_CHECKING:
    from guard.models import SecurityConfig

    from app.config.settings import Settings

configure_langsmith()
setup_logging()

if get_settings().OTEL_ENABLED:
    setup_otel(service_name="langchain-fastapi")


def create_app() -> FastAPI:
    """Create and configure FastAPI application with proper middleware order."""

    settings: Settings = get_settings()
    guard_config: SecurityConfig = build_fastapi_guard_config(settings)

    app: FastAPI = FastAPI(
        title="Langchain FastAPI Server",
        version=settings.APP_VERSION,
        lifespan=lifespan,
        docs_url="/api-docs",
        redoc_url="/api-redoc",
        openapi_url="/swagger.json",
    )

    # ============================================================================
    # Add middlewares in REVERSE order of execution
    # Last added = First executed
    #
    # Execution order (outermost → innermost):
    #   1. RequestStateLoggingMiddleware — correlation ID, request state
    #   2. SecurityMiddleware (Guard) — IP checks, rate limiting, pen-test detection, CORS headers
    #   3. GZipMiddleware — response compression
    #   4. ApiDeprecationMiddleware — Deprecation/Sunset headers on v1
    #   5. CORSMiddleware (injected by SecurityMiddleware.configure_cors())
    #      Guard's SecurityMiddleware deduplicates CORS headers internally.
    #      Do NOT add another CORSMiddleware directly.
    #   6. Route handler
    # ============================================================================

    # 1. CORS (managed by FastAPI Guard's helper — injects CORSMiddleware internally)
    SecurityMiddleware.configure_cors(app=app, config=guard_config)

    # 3. Compression (Performance optimization)
    app.add_middleware(GZipMiddleware, minimum_size=15000, compresslevel=6)

    # 4. API deprecation headers (v1 → v2 migration support)
    app.add_middleware(
        ApiDeprecationMiddleware,
        sunset_date=settings.API_SUNSET_DATE,
        v2_base_path=settings.API_V2_BASE_PATH,
    )

    # 5. Security middleware (headers, rate limiting, penetration detection)
    app.add_middleware(SecurityMiddleware, config=guard_config)

    # 6. Request state logging (Keep tracing context alive for streaming responses)
    app.add_middleware(RequestStateLoggingMiddleware)

    # 7. OTel ASGI instrumentation (auto-creates HTTP server spans)
    if settings.OTEL_ENABLED:
        app.add_middleware(
            OpenTelemetryMiddleware,
            default_span_details=default_span_details,
            excluded_urls="healthz,readyz,metrics",
        )

    # ============================================================================
    # EXCEPTION HANDLERS (Register after middleware, before routes)
    #
    # One call, four registrations, and every one of them is required: Starlette resolves
    # non-`Exception` handlers by walking the raised class's MRO, and FastAPI has already
    # `setdefault`-ed its own entries for `HTTPException` and `RequestValidationError`. The
    # single `add_exception_handler(Exception, ...)` that used to live on this line therefore
    # left three of `global_exception_handler`'s four branches unreachable and the documented
    # error envelope absent from every response except an unhandled 500. See
    # `register_exception_handlers` for the full mechanism before editing this.
    # ============================================================================
    register_exception_handlers(app)
    app.include_router(deep_health_router)

    # ============================================================================
    # ROUTES
    # ============================================================================

    @app.get(path="/", tags=["Root"])
    async def root(response: Response) -> APIResponse[dict[str, str]]:
        """Root endpoint - health check."""
        return render_result(
            Success(
                inner_value={
                    "message": "Root Route🚀",
                    "status": "healthy",
                    "version": settings.APP_VERSION,
                    "git_sha": settings.GIT_SHA,
                    "build_date": settings.BUILD_DATE,
                }
            ),
            response,
            message="Root route retrieved",
        )

    @app.get(path="/metrics", tags=["Monitoring"])
    async def metrics() -> Response:
        """Prometheus metrics endpoint — served via OTel PrometheusMetricReader."""
        data, content_type = get_metrics()
        return Response(content=data, media_type=content_type)

    # Include feature routers
    app.include_router(router=v1_router)
    app.include_router(router=v2_router)

    # 404 handler (Catch-all route)
    @app.api_route(
        path="/{path_name:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        include_in_schema=False,
        response_model=APIResponse[None],
        status_code=status.HTTP_404_NOT_FOUND,
    )
    async def catch_all(request: Request, path_name: str, response: Response) -> APIResponse[None]:
        """Handle 404 errors for undefined routes."""
        correlation_id = getattr(request.state, "correlation_id", "unknown")
        logger.bind(
            correlation_id=correlation_id,
            method=request.method,
            path=request.url.path,
            path_name=path_name,
        ).warning("404 Not Found")

        return render_result(
            Failure(
                inner_value=NotFoundError(
                    message=f"Can't find {request.url.path} on this server",
                    details={
                        "path": request.url.path,
                        "correlation_id": correlation_id,
                    },
                )
            ),
            response,
        )

    return app


app: FastAPI = create_app()
