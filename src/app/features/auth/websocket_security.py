from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastapi import WebSocket, WebSocketException, status
from fastapi_limiter.depends import WebSocketRateLimiter
from pyrate_limiter import Limiter, Rate, RedisBucket

if TYPE_CHECKING:
    from redis.asyncio import Redis

    from app.config import Settings
    from app.features.auth.security import TokenClaims

_USER_CONNECTIONS_KEY = "ws:user_connections:{}"
_SESSION_CONNECTIONS_KEY = "ws:session_connections:{}"
_CONNECTION_KEY = "ws:connection:{}"


@dataclass(frozen=True)
class WebSocketSecurityContext:
    claims: TokenClaims
    user_id: str
    session_id: str | None
    connection_id: str
    origin: str | None
    user_rate_limit_key: str
    connection_rate_limit_key: str


class WebSocketSecurityViolation(Exception):
    def __init__(
        self,
        *,
        error_code: str,
        message: str,
        close_code: int = status.WS_1008_POLICY_VIOLATION,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.close_code = close_code
        self.retryable = retryable


class WebSocketRateLimitExceeded(WebSocketSecurityViolation):
    def __init__(self) -> None:
        super().__init__(
            error_code="RATE_LIMIT_EXCEEDED",
            message="Rate limit exceeded. Please slow down.",
            retryable=True,
        )


class WebSocketIdleTimeout(WebSocketSecurityViolation):
    def __init__(self) -> None:
        super().__init__(
            error_code="IDLE_TIMEOUT",
            message="Connection closed due to inactivity.",
            retryable=True,
        )


async def _websocket_rate_identifier(websocket: WebSocket) -> str:
    return getattr(websocket.state, "ws_rate_limit_id", "ws:anonymous")


def _raise_websocket_rate_limit(*_: object, **__: object) -> None:
    raise WebSocketRateLimitExceeded()


def build_websocket_security_service(
    redis: Redis,
    settings: Settings,
) -> WebSocketSecurityService:
    user_bucket = RedisBucket.init(
        [Rate(settings.WEBSOCKET_USER_MESSAGE_RATE, settings.WEBSOCKET_USER_MESSAGE_PERIOD_SECONDS)],
        redis,
        "ws:user:messages",
    )
    connection_bucket = RedisBucket.init(
        [
            Rate(
                settings.WEBSOCKET_CONNECTION_MESSAGE_RATE,
                settings.WEBSOCKET_CONNECTION_MESSAGE_PERIOD_SECONDS,
            )
        ],
        redis,
        "ws:connection:messages",
    )
    user_limiter = WebSocketRateLimiter(
        limiter=Limiter(user_bucket),
        identifier=_websocket_rate_identifier,
        callback=_raise_websocket_rate_limit,
    )
    connection_limiter = WebSocketRateLimiter(
        limiter=Limiter(connection_bucket),
        identifier=_websocket_rate_identifier,
        callback=_raise_websocket_rate_limit,
    )
    return WebSocketSecurityService(
        redis=redis,
        settings=settings,
        user_limiter=user_limiter,
        connection_limiter=connection_limiter,
    )


class WebSocketSecurityService:
    def __init__(
        self,
        *,
        redis: Redis,
        settings: Settings,
        user_limiter: WebSocketRateLimiter,
        connection_limiter: WebSocketRateLimiter,
    ) -> None:
        self._redis = redis
        self._settings = settings
        self._user_limiter = user_limiter
        self._connection_limiter = connection_limiter

    def ensure_origin_allowed(self, origin: str | None) -> None:
        allowed_origins = self._settings.WEBSOCKET_ALLOWED_ORIGINS or [self._settings.FRONTEND_URL]
        if origin is None:
            if self._settings.WEBSOCKET_REQUIRE_ORIGIN:
                raise WebSocketException(
                    code=status.WS_1008_POLICY_VIOLATION,
                    reason="Missing Origin header",
                )
            return

        if origin not in allowed_origins:
            raise WebSocketException(
                code=status.WS_1008_POLICY_VIOLATION,
                reason="Origin not allowed",
            )

    def build_context(self, *, claims: TokenClaims, origin: str | None, connection_id: str) -> WebSocketSecurityContext:
        return WebSocketSecurityContext(
            claims=claims,
            user_id=claims.sub,
            session_id=claims.sid,
            connection_id=connection_id,
            origin=origin,
            user_rate_limit_key=f"user:{claims.sub}",
            connection_rate_limit_key=f"connection:{connection_id}",
        )

    async def ensure_connection_capacity(self, user_id: str) -> None:
        active_connections = await self.get_active_connection_count(user_id)
        if active_connections >= self._settings.WEBSOCKET_MAX_CONNECTIONS_PER_USER:
            raise WebSocketException(
                code=status.WS_1008_POLICY_VIOLATION,
                reason="Maximum concurrent WebSocket connections exceeded",
            )

    async def register_connection(self, context: WebSocketSecurityContext) -> None:
        connection_key = _CONNECTION_KEY.format(context.connection_id)
        user_key = _USER_CONNECTIONS_KEY.format(context.user_id)
        ttl = self._settings.WEBSOCKET_PRESENCE_TTL_SECONDS
        payload = json.dumps(
            {
                "user_id": context.user_id,
                "session_id": context.session_id,
            }
        )

        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.setex(connection_key, ttl, payload)
            pipe.sadd(user_key, context.connection_id)
            pipe.expire(user_key, ttl)
            if context.session_id is not None:
                session_key = _SESSION_CONNECTIONS_KEY.format(context.session_id)
                pipe.sadd(session_key, context.connection_id)
                pipe.expire(session_key, ttl)
            await pipe.execute()

    async def unregister_connection(self, context: WebSocketSecurityContext) -> None:
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.delete(_CONNECTION_KEY.format(context.connection_id))
            pipe.srem(_USER_CONNECTIONS_KEY.format(context.user_id), context.connection_id)
            if context.session_id is not None:
                pipe.srem(
                    _SESSION_CONNECTIONS_KEY.format(context.session_id),
                    context.connection_id,
                )
            await pipe.execute()

    async def touch_connection(self, context: WebSocketSecurityContext) -> None:
        ttl = self._settings.WEBSOCKET_PRESENCE_TTL_SECONDS
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.expire(_CONNECTION_KEY.format(context.connection_id), ttl)
            pipe.expire(_USER_CONNECTIONS_KEY.format(context.user_id), ttl)
            if context.session_id is not None:
                pipe.expire(_SESSION_CONNECTIONS_KEY.format(context.session_id), ttl)
            await pipe.execute()

    async def get_active_connection_count(self, user_id: str) -> int:
        user_key = _USER_CONNECTIONS_KEY.format(user_id)
        connection_ids = await self._redis.smembers(user_key)
        if not connection_ids:
            return 0

        connection_list = list(connection_ids)
        async with self._redis.pipeline(transaction=False) as pipe:
            for connection_id in connection_list:
                pipe.exists(_CONNECTION_KEY.format(connection_id))
            exists_results: list[int] = await pipe.execute()

        stale_connections = [
            connection_id
            for connection_id, exists in zip(connection_list, exists_results, strict=False)
            if not exists
        ]
        if stale_connections:
            await self._redis.srem(user_key, *stale_connections)

        return len(connection_list) - len(stale_connections)

    async def receive_json(
        self,
        websocket: WebSocket,
        context: WebSocketSecurityContext,
    ) -> object:
        try:
            payload = await asyncio.wait_for(
                websocket.receive_json(),
                timeout=self._settings.WEBSOCKET_IDLE_TIMEOUT_SECONDS,
            )
        except TimeoutError as exc:
            raise WebSocketIdleTimeout() from exc

        await self._apply_rate_limits(websocket, context)
        await self.touch_connection(context)
        return payload

    async def send_json(
        self,
        websocket: WebSocket,
        payload: object,
        context: WebSocketSecurityContext,
    ) -> None:
        await websocket.send_json(payload)
        await self.touch_connection(context)

    async def close_with_violation(
        self,
        websocket: WebSocket,
        context: WebSocketSecurityContext,
        violation: WebSocketSecurityViolation,
    ) -> None:
        with suppress(Exception):
            await self.send_json(
                websocket,
                {
                    "type": "error",
                    "node": None,
                    "code": violation.error_code,
                    "message": violation.message,
                    "retryable": violation.retryable,
                },
                context,
            )
        with suppress(Exception):
            await websocket.close(code=violation.close_code, reason=violation.message)

    async def _apply_rate_limits(
        self,
        websocket: WebSocket,
        context: WebSocketSecurityContext,
    ) -> None:
        websocket.state.ws_rate_limit_id = context.user_rate_limit_key
        await self._user_limiter(websocket, context_key="message")
        websocket.state.ws_rate_limit_id = context.connection_rate_limit_key
        await self._connection_limiter(websocket, context_key="message")
