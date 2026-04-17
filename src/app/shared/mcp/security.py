from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, cast

from nanoid import generate
from starlette.middleware import Middleware

from app.config import get_settings
from app.features.auth.security import decode_token
from app.middleware import observe_mcp_http_request
from app.utils import UnauthorizedException, logger
from app.utils.rate_limit.service import RateLimitService

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any


async def _send_json_response(
    send: Callable[[dict[str, Any]], Any],
    *,
    status_code: int,
    payload: dict[str, Any],
    headers: list[tuple[bytes, bytes]] | None = None,
) -> None:
    body = json.dumps(payload).encode("utf-8")
    response_headers = [(b"content-type", b"application/json")]
    if headers:
        response_headers.extend(headers)
    await send(
        {
            "type": "http.response.start",
            "status": status_code,
            "headers": response_headers,
        }
    )
    await send({"type": "http.response.body", "body": body, "more_body": False})


class MCPAuthMiddleware:
    def __init__(self, app: Any, enabled: bool = True) -> None:
        self.app = app
        self.enabled = enabled

    async def __call__(self, scope: dict[str, Any], receive: Callable, send: Callable) -> None:
        if scope["type"] != "http" or not self.enabled:
            await self.app(scope, receive, send)
            return

        authorization = ""
        for key, value in scope.get("headers", []):
            if key.lower() == b"authorization":
                authorization = value.decode()
                break

        if not authorization.startswith("Bearer "):
            exc = UnauthorizedException("Missing bearer token for MCP endpoint")
            await _send_json_response(
                send,
                status_code=exc.status_code,
                payload={"detail": exc.detail},
                headers=[(b"www-authenticate", b"Bearer")],
            )
            return

        token = authorization.removeprefix("Bearer ").strip()

        try:
            claims = decode_token(token)
        except UnauthorizedException as exc:
            await _send_json_response(
                send,
                status_code=exc.status_code,
                payload={"detail": exc.detail},
                headers=[(b"www-authenticate", b"Bearer")],
            )
            return

        state = scope.setdefault("state", {})
        state["mcp_claims"] = claims
        state["mcp_subject"] = claims.sub
        await self.app(scope, receive, send)


class MCPRateLimitMiddleware:
    def __init__(
        self,
        app: Any,
        redis_getter: Callable[[], Any | None],
        *,
        burst: int,
        rate: int,
        period_seconds: int,
    ) -> None:
        self.app = app
        self.redis_getter = redis_getter
        self.burst = burst
        self.rate = rate
        self.period_seconds = period_seconds

    async def __call__(self, scope: dict[str, Any], receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        redis = self.redis_getter()
        if redis is None:
            await self.app(scope, receive, send)
            return

        subject = scope.get("state", {}).get("mcp_subject") or _client_ip(scope)
        path = scope.get("path", "/mcp")
        key = f"mcp:{subject}:{path}"

        try:
            await RateLimitService(redis).check_limit(
                identifier=key,
                burst=self.burst,
                rate=self.rate,
                period_seconds=self.period_seconds,
            )
        except Exception as exc:
            status_code = getattr(exc, "status_code", 429)
            payload = getattr(exc, "detail", {"message": "Rate limit exceeded"})
            if not isinstance(payload, dict):
                payload = {"message": str(payload)}
            await _send_json_response(send, status_code=status_code, payload={"detail": payload})
            return

        await self.app(scope, receive, send)


class MCPObservabilityMiddleware:
    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        correlation_id = scope.setdefault("state", {}).get("correlation_id") or generate(size=21)
        scope["state"]["correlation_id"] = correlation_id
        path = scope.get("path", "/mcp")
        subject = scope.get("state", {}).get("mcp_subject", "anonymous")
        start = time.perf_counter()
        status_code = 500

        async def send_wrapper(message: dict[str, Any]) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = list(message.get("headers", []))
                headers.append((b"x-correlation-id", correlation_id.encode()))
                message["headers"] = headers
            await send(message)

        logger.bind(path=path, subject=subject, correlation_id=correlation_id).info("MCP request started")
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration = time.perf_counter() - start
            observe_mcp_http_request(path=path, status_code=status_code, duration_seconds=duration)
            logger.bind(
                path=path,
                subject=subject,
                correlation_id=correlation_id,
                status_code=status_code,
                duration_ms=round(duration * 1000, 2),
            ).info("MCP request completed")


def _client_ip(scope: dict[str, Any]) -> str:
    client = scope.get("client")
    if not client:
        return "unknown"
    return str(client[0])


def build_mcp_http_middleware(parent_app: Any | None) -> list[Middleware]:
    settings = get_settings()

    def get_redis() -> Any | None:
        if parent_app is None:
            return None
        return getattr(parent_app.state, "redis", None)

    return [
        Middleware(cast("Any", MCPObservabilityMiddleware)),
        Middleware(cast("Any", MCPAuthMiddleware), enabled=settings.MCP_REQUIRE_AUTH),
        Middleware(
            cast("Any", MCPRateLimitMiddleware),
            redis_getter=get_redis,
            burst=settings.MCP_RATE_LIMIT_BURST,
            rate=settings.MCP_RATE_LIMIT_RATE,
            period_seconds=settings.MCP_RATE_LIMIT_PERIOD_SECONDS,
        ),
    ]
