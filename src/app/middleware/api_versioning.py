"""Middleware to inject API deprecation headers on v1 routes.

Adds ``Deprecation``, ``Sunset``, and ``Link`` responses for
routes matching ``/api/v1/*``.  Exempt paths (health, metrics, docs) are
skipped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from starlette.middleware.base import BaseHTTPMiddleware

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from starlette.requests import Request
    from starlette.responses import Response

# Paths that should never get deprecation headers
_DEFAULT_EXEMPT_PREFIXES = frozenset(
    {"/health", "/metrics", "/api-docs", "/api-redoc", "/swagger.json"}
)


class ApiDeprecationMiddleware(BaseHTTPMiddleware):
    """Inject ``Deprecation``, ``Sunset``, and ``Link`` headers on v1 routes."""

    def __init__(
        self,
        app: object,
        *,
        sunset_date: str,
        v2_base_path: str = "/api/v2",
        exempt_prefixes: frozenset[str] | None = None,
    ) -> None:
        super().__init__(app)  # ty: ignore[invalid-argument-type]
        self._sunset_date = sunset_date
        self._v2_base_path = v2_base_path
        self._exempt = exempt_prefixes or _DEFAULT_EXEMPT_PREFIXES

    @override
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        path = request.url.path

        # Only apply to /api/v1/* routes
        if not path.startswith("/api/v1/"):
            return response

        # Skip exempt paths
        for prefix in self._exempt:
            if path.startswith(prefix):
                return response

        response.headers["Deprecation"] = "true"
        response.headers["Sunset"] = self._sunset_date
        response.headers["Link"] = (
            f'<{self._v2_base_path}{path[len("/api/v1") :]}>; rel="successor-version"'
        )
        return response
