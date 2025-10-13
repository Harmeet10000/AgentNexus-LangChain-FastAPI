"""Main FastAPI application module."""

from contextlib import asynccontextmanager
from typing import Any, Dict, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from app.utils.httpResponse import http_response
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import inspect

from app.core.cache import connect_to_redis
from app.core.settings import settings
from app.core.exceptions import register_exception_handlers
from app.db.mongodb import connect_to_mongodb
from app.middleware.server_middleware import CorrelationMiddleware
from app.middleware.security import SecurityHeadersMiddleware
from app.middleware.timeout import TimeoutMiddleware
from app.utils.logging import logger, setup_logging
from app.features.health.router import router as health_router


# Rate limiter
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan."""
    setup_logging()
    logger.info("Application starting")

    # Connect to databases
    await connect_to_mongodb()
    await connect_to_redis()

    yield

    logger.info("Application shutdown")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""

    app = FastAPI(
        title="LangChain FastAPI Production",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/api-docs",
        openapi_url="/swagger.json",
    )

    # Security middleware (first)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

    # Compression (15KB threshold like Express)
    app.add_middleware(GZipMiddleware, minimum_size=15000, compresslevel=6)

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Timeout (30 seconds)
    app.add_middleware(TimeoutMiddleware, timeout=30)

    # Correlation ID
    app.add_middleware(CorrelationMiddleware)

    # Rate limiter
    app.state.limiter = limiter

    # slowapi's handler expects a specific exception type; wrap it in an
    # async handler that accepts a generic Exception to satisfy FastAPI's
    # typing expectations while forwarding to slowapi's implementation.
    async def _rate_limit_handler_wrapper(request: Request, exc: Exception) -> Any:
        if not isinstance(exc, RateLimitExceeded):
            # If it's not the expected exception, return a generic 429 response.
            return JSONResponse(
                status_code=429, content={"detail": "Too Many Requests"}
            )

        result = _rate_limit_exceeded_handler(request, exc)
        # support either sync or async implementations
        if inspect.isawaitable(result):
            return await result
        return result

    app.add_exception_handler(RateLimitExceeded, _rate_limit_handler_wrapper)

    # Exception handlers
    register_exception_handlers(app)

    # Root endpoint
    @app.get("/")
    async def root() -> Dict[str, str]:
        return {"message": "Welcome to LangChain FastAPI Production 🚀"}

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=5000, reload=True)
