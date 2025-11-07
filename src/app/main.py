import sys
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from app.core.lifespan import lifespan
from app.core.settings import get_settings
from app.core.signals import setup_signal_handlers
from app.features.health.router import router as health_router
from app.middleware.server_middleware import (
    CorrelationMiddleware,
    MetricsMiddleware,
    TimeoutMiddleware,
    get_metrics,
)
from app.middleware.global_exception_handler import global_exception_handler
from app.utils.httpResponse import http_response
from app.utils.logger import logger, setup_logging

load_dotenv(".env.development")

# Initialize logging
# setup_logging()


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""

    app = FastAPI(
        title="LangChain FastAPI Production",
        version="1.0.0",
        lifespan=lifespan,
        docs_url=None if get_settings().ENVIRONMENT else "/api-docs",
        redoc_url=None if get_settings().ENVIRONMENT else "/api-redoc",
        openapi_url=None if get_settings().ENVIRONMENT else "/swagger.json",
    )

    # Correlation ID
    app.add_middleware(CorrelationMiddleware)

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

    # Root endpoint
    @app.get("/")
    async def root() -> dict[str, str]:
        return {"message": "Welcome to LangChain FastAPI  🚀"}

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


# Register GLOBAL exception handlers (equivalent to Express error middleware)
app.add_exception_handler(Exception, global_exception_handler)

if __name__ == "__main__":
    # Setup signal handlers
    setup_signal_handlers()

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        log_config=None,
        access_log=False,
    )
