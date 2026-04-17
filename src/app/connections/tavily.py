"""Tavily search client initialization and dependency injection."""

from functools import lru_cache

import httpx
from fastapi.requests import HTTPConnection

from app.config import get_settings


@lru_cache(maxsize=1)
def get_shared_tavily_http_client() -> httpx.AsyncClient:
    """Return a process-wide async HTTPX client for Tavily requests.

    Used for non-request runtimes like background tasks and Celery workers.
    """
    settings = get_settings()
    return httpx.AsyncClient(
        base_url=settings.TAVILY_BASE_URL.rstrip("/"),
        timeout=httpx.Timeout(settings.TAVILY_TIMEOUT_SECONDS),
        http2=True,
        limits=httpx.Limits(
            max_connections=20,
            max_keepalive_connections=5,
            keepalive_expiry=180.0,
        ),
        headers={
            "User-Agent": f"{settings.APP_NAME}/1.0",
        },
    )


def get_tavily_http_client(connection: HTTPConnection) -> httpx.AsyncClient:
    """Dependency to inject Tavily HTTP client from request lifespan."""
    return connection.app.state.tavily_http_client


async def create_tavily_http_client() -> httpx.AsyncClient:
    """Create Tavily HTTP client for FastAPI lifespan management."""
    return get_shared_tavily_http_client()


async def close_tavily_http_client(client: httpx.AsyncClient) -> None:
    """Close the Tavily HTTP client during lifespan shutdown."""
    await client.aclose()
