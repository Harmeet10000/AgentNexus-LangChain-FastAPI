from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import ORJSONResponse, Response

from app.config.settings import get_settings
from app.features.auth.router import router as auth_router
from app.features.health.router import router as health_router
from app.lifecycle.lifespan import lifespan
from app.middleware.global_exception_handler import global_exception_handler
from app.middleware.server_middleware import (
    MetricsMiddleware,
    TimeoutMiddleware,
    correlation_middleware,
    create_security_headers_middleware,
    get_metrics,
)
from app.utils.logger import logger

# Load environment variables
load_dotenv(dotenv_path=".env.development")


def create_app() -> FastAPI:
    """Create and configure FastAPI application with proper middleware order."""

    settings = get_settings()

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

    # 1. CORS (First to execute - handles preflight requests)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-Correlation-ID"],
        expose_headers=["X-Total-Count", "X-Correlation-ID", "X-Process-Time"],
        max_age=3600,
    )

    # 2. Trusted hosts (Security)
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.CORS_ORIGINS,
    )

    # 3. Compression (Performance optimization)
    app.add_middleware(GZipMiddleware, minimum_size=15000, compresslevel=6)

    # 4. Timeout (Prevent hanging requests)
    app.add_middleware(TimeoutMiddleware, timeout_seconds=30)

    # 5. Metrics collection (Monitor all requests)
    app.add_middleware(MetricsMiddleware, project_name="langchain-fastapi")

    # ============================================================================
    # CUSTOM MIDDLEWARES (Using decorator style for better performance)
    # ============================================================================

    # 6. Security headers (Execute early)
    @app.middleware("http")
    async def add_security_headers(request, call_next):
        return await create_security_headers_middleware(request, call_next)

    # 7. Correlation ID (For distributed tracing)
    @app.middleware("http")
    async def add_correlation_id(request: Request, call_next):
        return await correlation_middleware(request, call_next)

    # ============================================================================
    # EXCEPTION HANDLERS (Register after middleware, before routes)
    # ============================================================================
    app.add_exception_handler(Exception, global_exception_handler)

    # ============================================================================
    # ROUTES
    # ============================================================================

    @app.get("/", tags=["Root"])
    async def root() -> dict[str, str]:
        """Root endpoint - health check."""
        return {
            "message": "Root Route🚀",
            "status": "healthy",
            "version": "1.0.0",
        }

    @app.get("/metrics", tags=["Monitoring"])
    async def metrics() -> Response:
        """Prometheus metrics endpoint."""
        data, content_type = get_metrics()
        return Response(content=data, media_type=content_type)

    # Include feature routers
    app.include_router(health_router)
    app.include_router(auth_router)

    # 404 handler (Catch-all route)
    @app.api_route(
        "/{path_name:path}",
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
