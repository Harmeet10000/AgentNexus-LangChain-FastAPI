"""HTTPX client with optimal performance settings."""
from functools import lru_cache

import httpx
from fastapi.requests import HTTPConnection

from app.config import get_settings


def create_httpx_client() -> httpx.AsyncClient:
    """
    Create production-grade HTTPX client with HTTP/2 and connection pooling.

    Key optimizations:
    - HTTP/2 multiplexing (multiple requests over one connection)
    - Connection pooling (reuse TCP connections)
    - Automatic retries with exponential backoff
    - Request/response compression
    """
    settings = get_settings()

    return httpx.AsyncClient(
        # HTTP/2 for multiplexing (THE secret weapon)
        http2=True,
        # Connection pooling
        limits=httpx.Limits(
            max_connections=100,  # Total connections
            max_keepalive_connections=20,  # Persistent connections
            keepalive_expiry=180.0,  # Keep alive for 180s
        ),
        # Timeouts (prevent hanging)
        timeout=httpx.Timeout(
            connect=5.0,  # Connection timeout
            read=60.0,  # Read timeout
            write=10.0,  # Write timeout
            pool=5.0,  # Pool acquisition timeout
        ),
        # Automatic retries
        transport=httpx.AsyncHTTPTransport(
            retries=3,
            http2=True,
        ),
        # Headers
        headers={
            "User-Agent": f"{settings.APP_NAME}/1.0",
            "Accept-Encoding": "gzip, deflate, br",  # Compression
        },
        # Follow redirects
        follow_redirects=True,
        max_redirects=3,
    )


@lru_cache(maxsize=1)
def get_shared_httpx_client() -> httpx.AsyncClient:
    """Return a process-wide async HTTPX client for non-request runtimes."""
    return create_httpx_client()


def get_httpx_client(connection: HTTPConnection) -> httpx.AsyncClient:
    """Dependency to inject HTTPX client."""
    return connection.app.state.httpx_client
