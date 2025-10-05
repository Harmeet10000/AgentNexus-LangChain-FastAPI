"""Main FastAPI application module."""

from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from api.middleware.error_handler import ErrorHandlerMiddleware
from api.middleware.logging_middleware import LoggingMiddleware
from api.middleware.rate_limiter import RateLimiterMiddleware
from api.middleware.correlation_id import CorrelationIDMiddleware
from api.router import api_router
from core.config.logging_config import LoggerAdapter
from core.config.settings import settings
from core.database.mongodb import close_mongodb_connection, connect_to_mongodb
from core.cache.redis_client import close_redis_connection, connect_to_redis
from services.langsmith.client import initialize_langsmith
from services.pinecone.client import initialize_pinecone


logger = LoggerAdapter(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting application", environment=settings.environment)
    
    try:
        # Initialize connections
        await connect_to_mongodb()
        await connect_to_redis()
        
        # Initialize services
        if not settings.is_testing:
            initialize_pinecone()
            initialize_langsmith()
        
        logger.info("Application started successfully")
        
        yield
        
    finally:
        # Cleanup connections
        await close_mongodb_connection()
        await close_redis_connection()
        
        logger.info("Application shutdown complete")


def create_application() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        lifespan=lifespan,
        docs_url=f"{settings.api_prefix}/docs" if not settings.is_production else None,
        redoc_url=f"{settings.api_prefix}/redoc" if not settings.is_production else None,
        openapi_url=f"{settings.api_prefix}/openapi.json" if not settings.is_production else None,
    )
    
    # Add middleware
    setup_middleware(app)
    
    # Include routers
    app.include_router(api_router, prefix=settings.api_prefix)
    
    # Add health check endpoint
    @app.get("/health")
    async def health_check() -> Dict[str, Any]:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "environment": settings.environment,
            "version": settings.app_version,
        }
    
    # Add root endpoint
    @app.get("/")
    async def root() -> Dict[str, str]:
        """Root endpoint."""
        return {
            "message": "LangChain FastAPI Production Server",
            "docs": f"{settings.api_prefix}/docs",
            "health": "/health",
        }
    
    return app


def setup_middleware(app: FastAPI) -> None:
    """Configure application middleware."""
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Trusted host middleware
    if settings.is_production:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"],  # Configure based on your domain
        )
    
    # Custom middleware (order matters - first added is outermost)
    app.add_middleware(ErrorHandlerMiddleware)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(CorrelationIDMiddleware)  # Add correlation ID middleware
    
    if settings.rate_limit_enabled:
        app.add_middleware(RateLimiterMiddleware)


# Create application instance
app = create_application()


# Exception handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={"detail": "Resource not found"},
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors."""
    logger.error("Internal server error", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.is_development,
        workers=1 if settings.is_development else settings.workers,
        log_level=settings.log_level.lower(),
    )