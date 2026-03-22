from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import ORJSONResponse, Response
from fastmcp.utilities.lifespan import combine_lifespans
from guard import SecurityMiddleware

from app.api.v1 import v1_router
from app.api.v2 import v2_router
from app.config import get_settings
from app.lifecycle import lifespan
from app.middleware import (
    MetricsMiddleware,
    RequestStateLoggingMiddleware,
    build_fastapi_guard_config,
    get_metrics,
    global_exception_handler,
)
from app.shared.langchain_layer import configure_langsmith
from app.shared.mcp import get_mcp_http_app
from app.shared.mcp.models import parse_mcp_http_transport
from app.utils import logger

configure_langsmith()
# Load environment variables
load_dotenv(dotenv_path=".env.development")


def create_app() -> FastAPI:
    """Create and configure FastAPI application with proper middleware order."""

    settings = get_settings()
    guard_config = build_fastapi_guard_config(settings)

    app: FastAPI = FastAPI(
        title="Langchain FastAPI Template",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/api-docs",
        redoc_url="/api-redoc",
        openapi_url="/swagger.json",
        default_response_class=ORJSONResponse,
    )

    # ============================================================================
    # Add middlewares in REVERSE order of execution
    # Last added = First executed
    # ============================================================================

    # 1. CORS (managed by FastAPI Guard's helper)
    SecurityMiddleware.configure_cors(app=app, config=guard_config)

    # 3. Compression (Performance optimization)
    app.add_middleware(GZipMiddleware, minimum_size=15000, compresslevel=6)  # ty:ignore[invalid-argument-type]

    # 4. Timeout (Prevent hanging requests)
    # app.add_middleware(TimeoutMiddleware, timeout_seconds=30)

    # 5. Security middleware (headers, rate limiting, penetration detection)
    app.add_middleware(SecurityMiddleware, config=guard_config)  # ty:ignore[invalid-argument-type]

    # 6. Metrics collection (Monitor requests, including guard-blocked traffic)
    app.add_middleware(MetricsMiddleware, project_name="langchain-fastapi")  # ty:ignore[invalid-argument-type]

    # 7. Request state logging (Keep tracing context alive for streaming responses)
    app.add_middleware(RequestStateLoggingMiddleware)  # ty:ignore[invalid-argument-type]

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
    )
    async def catch_all(request: Request, path_name: str) -> ORJSONResponse:
        """Handle 404 errors for undefined routes."""
        correlation_id = getattr(request.state, "correlation_id", "unknown")
        logger.warning(
            f"[{correlation_id}] 404 Not Found: {request.method} {request.url.path} {path_name}"
        )

        return ORJSONResponse(
            status_code=404,
            content={
                "error": "Not Found",
                "message": f"Can't find {request.url.path} on this server",
                "path": request.url.path,
                "correlation_id": correlation_id,
            },
        )

    return app


app: FastAPI = create_app()
