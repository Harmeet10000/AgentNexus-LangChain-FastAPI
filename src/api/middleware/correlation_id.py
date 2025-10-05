"""Correlation ID middleware."""

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from shared.utils.id_generator import generate_correlation_id


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """Middleware to add correlation ID to each request."""

    async def dispatch(self, request: Request, call_next):
        """Add correlation ID to request and response."""
        # Get correlation ID from header or generate new one using nanoid
        correlation_id = request.headers.get('X-Correlation-ID', generate_correlation_id())

        # Store in request state
        request.state.correlation_id = correlation_id

        # Process request
        response = await call_next(request)

        # Add correlation ID to response headers
        response.headers['X-Correlation-ID'] = correlation_id

        return response