"""Request timeout middleware."""
import asyncio
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from app.utils.httpError import http_error
from app.utils.logging import logger
from app.middleware.correlational_middleware import get_correlation_id


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Timeout requests after specified duration."""

    def __init__(self, app, timeout: int = 30):
        super().__init__(app)
        self.timeout = timeout

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await asyncio.wait_for(call_next(request), timeout=self.timeout)
        except asyncio.TimeoutError:
            correlation_id = get_correlation_id()
            logger.warn("Request timed out", {
                "correlationId": correlation_id,
                "url": str(request.url),
                "method": request.method
            })
            exc = http_error(
                "Request took too long to process",
                408,
                request,
                Exception("Timeout")
            )
            return Response(
                content=str(exc.detail),
                status_code=exc.status_code,
                media_type="application/json"
            )
