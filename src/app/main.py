"""Main FastAPI application module."""
from dotenv import load_dotenv

# Load the specific environment file
load_dotenv(".env.development")
from contextlib import asynccontextmanager
from typing import Any, Dict, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from app.utils.httpResponse import http_response
import inspect

from app.core.cache import connect_to_redis
from app.core.settings import Settings, get_settings
from app.core.exceptions import register_exception_handlers
from app.connections.mongodb import connect_to_mongodb
from app.connections.postgres import init_db
from app.middleware.server_middleware import (
    CorrelationMiddleware,
    MetricsMiddleware,
    SecurityHeadersMiddleware,
    TimeoutMiddleware,
    init_rate_limiter,
    rate_limiter,
    get_metrics,
)
from app.utils.logger import logger, setup_logging
from app.features.health.router import router as health_router
import uvicorn
import sys
from app.middleware.server_middleware import MetricsMiddleware, get_metrics


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan."""
    setup_logging()
    logger.info("Application starting")

    # Connect to databases
    await connect_to_mongodb()
    # await connect_to_redis()
    # await init_db()

    yield

    logger.info("Application shutdown")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""

    app = FastAPI(
        title="LangChain FastAPI Production",
        version="1.0.0",
        lifespan=lifespan,
        docs_url=None if get_settings().ENVIRONMENT else  "/api-docs",
        redoc_url=None if get_settings().ENVIRONMENT else "/api-redoc",
        openapi_url=None if get_settings().ENVIRONMENT else "/swagger.json",
    )

    # Correlation ID
    app.add_middleware(CorrelationMiddleware)

    # Security middleware (first)
    app.add_middleware(SecurityHeadersMiddleware)
    # Trusted hosts
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

    # Compression (15KB threshold like Express)
    app.add_middleware(GZipMiddleware, minimum_size=15000, compresslevel=6)

    # Metrics
    app.add_middleware(MetricsMiddleware, project_name="langchain-fastapi")
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
        expose_headers=["X-Total-Count"],
        max_age=3600,
    )

    # Timeout (30 seconds)
    app.add_middleware(TimeoutMiddleware, timeout=30)

    # Exception handlers
    register_exception_handlers(app)

    # Root endpoint
    @app.get("/")
    async def root() -> Dict[str, str]:
        return {"message": "Welcome to LangChain FastAPI Production 🚀"}

    @app.get("/metrics")
    async def metrics(request: Request):
        data, content_type = get_metrics()
        return http_response(
            message="Success",
            data=data,
            status_code=200,
            request=request,
        )

    app.include_router(health_router)

    # 404 handler
    @app.api_route(
        "/{path_name:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"]
    )
    async def catch_all(request: Request, path_name: str) -> JSONResponse:
        return http_response(
            message=f"Can't find {request.url.path} on this server!",
            data=None,
            status_code=404,
            request=request,
        )

    return app


app = create_app()


def setup_signal_handlers() -> None:
    """Setup graceful shutdown handlers."""
    import signal
    import sys
    from types import FrameType

    def graceful_shutdown(sig_name: str) -> None:
        """Handle graceful shutdown."""
        logger.info(f"Received {sig_name}, shutting down gracefully...")
        sys.exit(0)

    def handle_sigterm(signum: int, frame: FrameType | None) -> None:
        graceful_shutdown("SIGTERM")

    def handle_sigint(signum: int, frame: FrameType | None) -> None:
        graceful_shutdown("SIGINT")

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigint)


if __name__ == "__main__":


    # Setup signal handlers
    setup_signal_handlers()

    # Handle unhandled exceptions
    def handle_exception(
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_traceback: Any
    ) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.error("UNHANDLED EXCEPTION! 💥", {"error": str(exc_value)})

    sys.excepthook = handle_exception

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        log_config=None  # Use loguru instead
    )
