import asyncio

from dotenv import load_dotenv
from fastapi import FastAPI, Request, status
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import Response
from fastmcp.utilities.lifespan import combine_lifespans
from guard import SecurityMiddleware

from .api import v1_router, v2_router
from .config import get_settings
from .lifecycle import lifespan
from .middleware import (
    # MetricsMiddleware,
    RequestStateLoggingMiddleware,
    build_fastapi_guard_config,
    get_metrics,
    global_exception_handler,
)
from .middleware.api_versioning import ApiDeprecationMiddleware
from .middleware.health_check import ALL_PROBES
from .shared.langchain_layer import configure_langsmith
from .shared.mcp import get_mcp_http_app, parse_mcp_http_transport
from .utils import APIResponse, ErrorCode, http_error, logger
from .utils.response_type import DependencyHealth, HealthResponse, HealthStatus

configure_langsmith()
# Load environment variables
load_dotenv(dotenv_path=".env.development")


def create_app() -> FastAPI:
    """Create and configure FastAPI application with proper middleware order."""

    settings = get_settings()
    guard_config = build_fastapi_guard_config(settings)

    app: FastAPI = FastAPI(
        title="Langchain FastAPI Server",
        version="1.0.0",
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
    #   2. SecurityMiddleware (Guard) — IP checks, rate limiting, pen-test detection
    #   3. GZipMiddleware — response compression
    #   4. ApiDeprecationMiddleware — Deprecation/Sunset headers on v1
    #   5. CORSMiddleware (via Guard.configure_cors) — CORS headers
    #   6. Route handler
    # ============================================================================

    # 1. CORS (managed by FastAPI Guard's helper)
    SecurityMiddleware.configure_cors(app=app, config=guard_config)

    # 3. Compression (Performance optimization)
    app.add_middleware(GZipMiddleware, minimum_size=15000, compresslevel=6)

    # 4. API deprecation headers (v1 → v2 migration support)
    app.add_middleware(
        ApiDeprecationMiddleware,
        sunset_date=settings.API_SUNSET_DATE,
        v2_base_path=settings.API_V2_BASE_PATH,
    )

    # 5. Timeout (Prevent hanging requests)
    # app.add_middleware(TimeoutMiddleware, timeout_seconds=30)

    # 5. Security middleware (headers, rate limiting, penetration detection)
    app.add_middleware(SecurityMiddleware, config=guard_config)

    # 6. Metrics collection (Monitor requests, including guard-blocked traffic)
    # app.add_middleware(MetricsMiddleware, project_name="langchain-fastapi")

    # 7. Request state logging (Keep tracing context alive for streaming responses)
    app.add_middleware(RequestStateLoggingMiddleware)

    # ============================================================================
    # EXCEPTION HANDLERS (Register after middleware, before routes)
    # ============================================================================
    app.add_exception_handler(exc_class_or_status_code=Exception, handler=global_exception_handler)

    # ============================================================================
    # ROUTES
    # ============================================================================

    @app.get(path="/", tags=["Root"])
    async def root() -> dict[str, str]:
        """Root endpoint - health check."""
        return {
            "message": "Root Route🚀",
            "status": "healthy",
            "version": "1.0.0",
        }

    @app.get(path="/health", tags=["Monitoring"])
    async def health() -> Response:
        """Deep health check — probes all critical dependencies in parallel."""
        results = await asyncio.gather(
            *[probe(app) for probe in ALL_PROBES],
            return_exceptions=True,
        )
        deps: list[DependencyHealth] = []
        for r in results:
            if isinstance(r, Exception):
                deps.append(DependencyHealth.fail("unknown", str(r)))
            else:
                deps.append(r)

        failed = sum(1 for d in deps if d.status == HealthStatus.UNHEALTHY)
        if failed >= 3:
            overall = HealthStatus.UNHEALTHY
        elif failed >= 1:
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.HEALTHY

        body = HealthResponse(status=overall, dependencies=deps)
        code = status.HTTP_503_SERVICE_UNAVAILABLE if overall == HealthStatus.UNHEALTHY else status.HTTP_200_OK
        return Response(
            content=body.model_dump_json(),
            status_code=code,
            media_type="application/json",
        )

    @app.get(path="/metrics", tags=["Monitoring"])
    async def metrics() -> Response:
        """Prometheus metrics endpoint."""
        data, content_type = get_metrics()
        return Response(content=data, media_type=content_type)

    # Include feature routers
    app.include_router(router=v1_router)
    app.include_router(router=v2_router)

    if settings.MCP_ENABLE_HTTP:
        mcp_transport = parse_mcp_http_transport(settings.MCP_HTTP_TRANSPORT)
        mcp_app = get_mcp_http_app(
            parent_app=app,
            path="/",
            transport=mcp_transport,
        )
        app.router.lifespan_context = combine_lifespans(lifespan, mcp_app.lifespan)
        app.mount(settings.MCP_HTTP_PATH, mcp_app)

    # 404 handler (Catch-all route)
    @app.api_route(
        path="/{path_name:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        include_in_schema=False,
        response_model=APIResponse[None],
        status_code=status.HTTP_404_NOT_FOUND,
    )
    async def catch_all(request: Request, path_name: str) -> APIResponse[None]:
        """Handle 404 errors for undefined routes."""
        correlation_id = getattr(request.state, "correlation_id", "unknown")
        logger.warning(
            f"[{correlation_id}] 404 Not Found: {request.method} {request.url.path} {path_name}"
        )

        return http_error(
            message=f"Can't find {request.url.path} on this server",
            status_code=status.HTTP_404_NOT_FOUND,
            error_code=ErrorCode.NOT_FOUND,
            data={
                "path": request.url.path,
                "correlation_id": correlation_id,
            },
        )

    return app


app: FastAPI = create_app()
