from collections.abc import Awaitable, Callable

from fastapi import Request
from pydantic import ValidationError

from app.utils.exceptions import UnauthorizedException

from .service import RateLimitService

_ACCESS_TOKEN_TYPE = "access"  # noqa: S105 — token type constant, not a password


def _extract_presented_token(request: Request) -> str | None:
    """Bearer header first, access-token cookie second; None when neither is present."""
    authorization = request.headers.get("authorization")
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token:
            return token
    cookie_token: str | None = request.cookies.get("access_token")
    return cookie_token or None


def _resolve_user_identifier(request: Request) -> str | None:
    """Subject of a valid access token, namespaced, or None when the request carries none.

    Identity comes from validated token claims, never from per-request state.
    Fail-open: an absent, expired, or malformed token — or a token of the wrong
    type — yields None so the caller falls back to IP keying instead of
    rejecting the request. A rate limiter must not become an auth gate,
    especially on the unauthenticated endpoints that use it.
    """
    raw_token = _extract_presented_token(request)
    if raw_token is None:
        return None
    # Deferred import: features/auth already imports this module for its
    # routers, so a top-level import would tie the two packages together.
    from app.features.auth.security import decode_token  # noqa: PLC0415

    try:
        claims = decode_token(raw_token)
    except (UnauthorizedException, ValidationError, KeyError):
        return None
    if claims.token_type != _ACCESS_TOKEN_TYPE:
        return None
    return f"user:{claims.sub}"


def get_rate_limiter(
    burst: int = 10, rate: int = 5, period: int = 60
) -> Callable[[Request], Awaitable[None]]:
    """
    Build a FastAPI dependency that enforces a Redis-backed rate limit.

    Identity is per-user when the request carries a valid access token
    (keyed ``user:<subject>``) and per-IP otherwise, so the limiter works
    unchanged on unauthenticated endpoints.

    Args:
        burst: Maximum burst capacity before requests are throttled.
        rate: Number of tokens refilled per period.
        period: Refill period in seconds.
    """

    async def rate_limit_dependency(request: Request) -> None:
        redis_client = request.app.state.redis
        forwarded_for = request.headers.get("X-Forwarded-For")
        client_ip = request.client.host if request.client else "unknown"
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip() or client_ip

        identifier = _resolve_user_identifier(request) or client_ip
        service = RateLimitService(redis_client)
        await service.check_limit(
            identifier=identifier, burst=burst, rate=rate, period_seconds=period
        )

    return rate_limit_dependency
