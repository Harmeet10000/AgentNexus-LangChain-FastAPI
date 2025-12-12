import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse,ORJSONResponse, Response

from app.core.lifespan import lifespan
from app.core.settings import get_settings
from app.core.signals import setup_signal_handlers
from app.features.health.router import router as health_router
from app.middleware.global_exception_handler import global_exception_handler
from app.middleware.server_middleware import (
    correlation_middleware,
    create_metrics_middleware,
    # create_security_headers_middleware,
    create_timeout_middleware,
    get_metrics,
)
from app.utils.logger import logger

# Load environment variables
load_dotenv(".env.development")


def create_app() -> FastAPI:
    """Create and configure FastAPI application with proper middleware order."""

    settings = get_settings()

    app = FastAPI(
        title="LangChain FastAPI Production",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/api-docs",
        redoc_url="/api-redoc",
        openapi_url="/swagger.json",
        default_response_class=ORJSONResponse
    )

    # ============================================================================
    # MIDDLEWARE ORDER (CRITICAL!)
    # Add middlewares in REVERSE order of execution
    # Last added = First executed
    # ============================================================================

    # 1. CORS (First to execute - handles preflight requests)
    # ✅ Fixed: Specific origins in production
    cors_origins = (
        ["*"]
        if settings.ENVIRONMENT != "production"
        else [
            "https://yourdomain.com",
            "https://app.yourdomain.com",
        ]
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        # Can't use allow_credentials=True with allow_origins=["*"]
        allow_credentials=cors_origins != ["*"],
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-Correlation-ID"],
        expose_headers=["X-Total-Count", "X-Correlation-ID", "X-Process-Time"],
        max_age=3600,
    )

    # 2. Trusted hosts (Security)
    allowed_hosts = (
        ["*"]
        if settings.ENVIRONMENT != "production"
        else [
            "yourdomain.com",
            "*.yourdomain.com",
            "localhost",
        ]
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

    # 3. Compression (Performance optimization)
    app.add_middleware(GZipMiddleware, minimum_size=15000, compresslevel=6)

    # ============================================================================
    # CUSTOM MIDDLEWARES (Using decorator style for better performance)
    # ============================================================================

    # # 4. Security headers (Execute early)
    # @app.middleware("http")
    # async def add_security_headers(request: Request, call_next):
    #     return await create_security_headers_middleware(request, call_next)

    # 5. Correlation ID (For distributed tracing)
    @app.middleware("http")
    async def add_correlation_id(request: Request, call_next):
        return await correlation_middleware(request, call_next)

    # 6. Metrics collection (Monitor all requests)
    metrics_middleware = create_metrics_middleware(project_name="langchain-fastapi")

    @app.middleware("http")
    async def collect_metrics(request: Request, call_next):
        return await metrics_middleware(request, call_next)

    # 7. Request timeout (Prevent hanging requests)
    @app.middleware("http")
    async def timeout_requests(request: Request, call_next):
        timeout_middleware = create_timeout_middleware(timeout_seconds=30)
        return await timeout_middleware(request, call_next)

    # 8. Error handling (Last middleware = catches all errors)
    @app.middleware("http")
    async def error_handler(request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as e:
            # Log the error
            correlation_id = getattr(request.state, "correlation_id", "unknown")
            logger.error(
                f"[{correlation_id}] Unhandled exception: {str(e)}", exc_info=True
            )

            # Return JSON error response
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "message": "An unexpected error occurred",
                    "correlation_id": correlation_id,
                },
            )

    # ============================================================================
    # ROUTES
    # ============================================================================

    @app.get("/", tags=["Root"])
    async def root() -> dict[str, str]:
        """Root endpoint - health check."""
        return {
            "message": "Welcome to LangChain FastAPI 🚀",
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

    # 404 handler (Catch-all route)
    @app.api_route(
        "/{path_name:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        include_in_schema=False,
    )
    async def catch_all(request: Request, path_name: str) -> JSONResponse:
        """Handle 404 errors for undefined routes."""
        correlation_id = getattr(request.state, "correlation_id", "unknown")
        logger.warning(
            f"[{correlation_id}] 404 Not Found: {request.method} {request.url.path} {path_name}"
        )

        return JSONResponse(
            status_code=404,
            content={
                "error": "Not Found",
                "message": f"Can't find {request.url.path} on this server",
                "path": request.url.path,
                "correlation_id": correlation_id,
            },
        )

    return app


app = create_app()


# ============================================================================
# GLOBAL EXCEPTION HANDLERS
# ============================================================================
app.add_exception_handler(Exception, global_exception_handler)


if __name__ == "__main__":
    # Setup signal handlers for graceful shutdown
    setup_signal_handlers()

    settings = get_settings()

    logger.info(f"Starting server in {settings.ENVIRONMENT} mode...")

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=5000,
        reload=settings.ENVIRONMENT != "production",
        log_config=None,  # Use custom logging
        access_log=False,  # Custom access logging via middleware
        # For production: use workers
        # workers=4 if settings.ENVIRONMENT == "production" else 1,
    )
