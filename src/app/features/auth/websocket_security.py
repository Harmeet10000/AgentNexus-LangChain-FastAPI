from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from datetime import UTC, datetime
from time import time
from typing import (
    TYPE_CHECKING,
)

from fastapi import WebSocket, WebSocketException, status
from pydantic import BaseModel, ConfigDict
from pyrate_limiter import BucketFullException, Limiter, Rate
from pyrate_limiter.buckets import InMemoryBucket, RedisBucket
from returns.result import Failure

from app.utils import logger

if TYPE_CHECKING:
    from redis.asyncio import Redis

    from app.config import Settings
    from app.features.auth.repository import RefreshTokenRepository
    from app.features.auth.security import TokenClaims

# Task 3.2: New sorted set key patterns
_USER_PRESENCE_KEY = "ws:user:{}"  # Sorted set: member=connection_id, score=last_touch_epoch
_SESSION_PRESENCE_KEY = "ws:session:{}"  # Sorted set: member=connection_id, score=last_touch_epoch
_CONNECTION_METADATA_KEY = "ws:connection:{}"  # Hash: connection metadata


class WebSocketSecurityContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    claims: TokenClaims
    user_id: str
    session_id: str | None
    connection_id: str
    origin: str | None
    user_rate_limit_key: str
    connection_rate_limit_key: str


class WebSocketSecurityViolationError(Exception):
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


WebSocketSecurityViolation = WebSocketSecurityViolationError


class WebSocketRateLimitExceededError(WebSocketSecurityViolationError):
    def __init__(self) -> None:
        super().__init__(
            error_code="RATE_LIMIT_EXCEEDED",
            message="Rate limit exceeded. Please slow down.",
            retryable=True,
        )


class WebSocketSessionRevokedError(WebSocketSecurityViolationError):
    def __init__(self) -> None:
        super().__init__(
            error_code="SESSION_REVOKED",
            message="Your session has been revoked. Please reconnect.",
            retryable=False,
        )


class WebSocketIdleTimeoutError(WebSocketSecurityViolationError):
    def __init__(self) -> None:
        super().__init__(
            error_code="IDLE_TIMEOUT",
            message="Connection closed due to inactivity.",
            retryable=True,
        )


WebSocketIdleTimeout = WebSocketIdleTimeoutError


def _raise_websocket_rate_limit(*_: object, **__: object) -> None:
    raise WebSocketRateLimitExceededError


async def build_websocket_security_service(
    redis: Redis | None,
    settings: Settings,
    token_repo: RefreshTokenRepository | None = None,
) -> WebSocketSecurityService:
    user_rates = [
        Rate(settings.WEBSOCKET_USER_MESSAGE_RATE, settings.WEBSOCKET_USER_MESSAGE_PERIOD_SECONDS)
    ]
    connection_rates = [
        Rate(
            settings.WEBSOCKET_CONNECTION_MESSAGE_RATE,
            settings.WEBSOCKET_CONNECTION_MESSAGE_PERIOD_SECONDS,
        )
    ]

    if redis is not None:
        try:
            user_bucket = await RedisBucket.init(user_rates, redis, "ws:user:messages")
            connection_bucket = await RedisBucket.init(
                connection_rates,
                redis,
                "ws:connection:messages",
            )
        except Exception as exc:  # noqa: BLE001 — fall back to in-memory rate limiter
            logger.warning(
                "Redis-based WebSocket rate limiter failed, falling back to in-memory bucket",
                error=str(exc),
            )
            user_bucket = InMemoryBucket(user_rates)
            connection_bucket = InMemoryBucket(connection_rates)
    else:
        user_bucket = InMemoryBucket(user_rates)
        connection_bucket = InMemoryBucket(connection_rates)

    # Task 3.3: Direct Limiter instances instead of WebSocketRateLimiter wrapper
    user_limiter = Limiter(user_bucket)
    connection_limiter = Limiter(connection_bucket)

    return WebSocketSecurityService(
        redis=redis,
        settings=settings,
        user_limiter=user_limiter,
        connection_limiter=connection_limiter,
        token_repo=token_repo,
    )


class WebSocketSecurityService:
    def __init__(
        self,
        *,
        redis: Redis | None,
        settings: Settings,
        user_limiter: Limiter,
        connection_limiter: Limiter,
        token_repo: RefreshTokenRepository | None = None,
    ) -> None:
        self._redis = redis
        self._settings = settings
        self._user_limiter = user_limiter
        self._connection_limiter = connection_limiter
        self._token_repo = token_repo
        # Task 3.2: Track last touch time per connection for throttling
        self._last_touch_time: dict[str, float] = {}

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

    @staticmethod
    def build_context(
        *, claims: TokenClaims, origin: str | None, connection_id: str
    ) -> WebSocketSecurityContext:
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
        """Task 3.2: Atomic capacity check using sorted sets."""
        if self._redis is None:
            return

        user_key = _USER_PRESENCE_KEY.format(user_id)
        current_epoch = time()
        ttl_seconds = self._settings.WEBSOCKET_PRESENCE_TTL_SECONDS
        cutoff_epoch = current_epoch - ttl_seconds

        # Atomic operation: evict stale + count
        async with self._redis.pipeline(transaction=True) as pipe:
            # Remove stale entries (older than TTL)
            pipe.zremrangebyscore(user_key, "-inf", cutoff_epoch)
            # Count remaining entries
            pipe.zcard(user_key)
            results = await pipe.execute()

        # results[1] is the count after eviction
        active_count = results[1] if len(results) > 1 else 0

        if active_count >= self._settings.WEBSOCKET_MAX_CONNECTIONS_PER_USER:
            raise WebSocketException(
                code=status.WS_1008_POLICY_VIOLATION,
                reason="Maximum concurrent WebSocket connections exceeded",
            )

    async def register_connection(self, context: WebSocketSecurityContext) -> None:
        """Task 3.2: Register connection in sorted sets."""
        if self._redis is None:
            return

        current_epoch = time()
        ttl = self._settings.WEBSOCKET_PRESENCE_TTL_SECONDS

        metadata = json.dumps(
            {
                "user_id": context.user_id,
                "session_id": context.session_id,
                "created_at": datetime.now(UTC).isoformat(),
            }
        )

        async with self._redis.pipeline(transaction=True) as pipe:
            # Add to user presence sorted set
            pipe.zadd(
                _USER_PRESENCE_KEY.format(context.user_id),
                {context.connection_id: current_epoch},
            )
            pipe.expire(_USER_PRESENCE_KEY.format(context.user_id), ttl)

            # Add to session presence sorted set (for revocation lookup)
            if context.session_id is not None:
                pipe.zadd(
                    _SESSION_PRESENCE_KEY.format(context.session_id),
                    {context.connection_id: current_epoch},
                )
                pipe.expire(_SESSION_PRESENCE_KEY.format(context.session_id), ttl)

            # Store connection metadata
            pipe.setex(_CONNECTION_METADATA_KEY.format(context.connection_id), ttl, metadata)

            await pipe.execute()

        # Initialize last touch time for throttling
        self._last_touch_time[context.connection_id] = current_epoch

    async def unregister_connection(self, context: WebSocketSecurityContext) -> None:
        """Task 3.2: Unregister connection from sorted sets."""
        if self._redis is None:
            return

        async with self._redis.pipeline(transaction=True) as pipe:
            # Remove from user presence sorted set
            pipe.zrem(_USER_PRESENCE_KEY.format(context.user_id), context.connection_id)

            # Remove from session presence sorted set
            if context.session_id is not None:
                pipe.zrem(
                    _SESSION_PRESENCE_KEY.format(context.session_id),
                    context.connection_id,
                )

            # Delete metadata
            pipe.delete(_CONNECTION_METADATA_KEY.format(context.connection_id))

            await pipe.execute()

        # Clean up touch time tracking
        self._last_touch_time.pop(context.connection_id, None)

    async def touch_connection(self, context: WebSocketSecurityContext) -> None:
        """Task 3.2: Update connection presence with throttling (Finding 1)."""
        if self._redis is None:
            return

        current_epoch = time()
        # Throttle: only touch if 30 seconds have elapsed since last touch
        last_touch = self._last_touch_time.get(context.connection_id, 0)
        if current_epoch - last_touch < 30:
            return

        self._last_touch_time[context.connection_id] = current_epoch
        ttl = self._settings.WEBSOCKET_PRESENCE_TTL_SECONDS

        async with self._redis.pipeline(transaction=True) as pipe:
            # Update score (last touch epoch) in user presence sorted set
            pipe.zadd(
                _USER_PRESENCE_KEY.format(context.user_id),
                {context.connection_id: current_epoch},
            )
            pipe.expire(_USER_PRESENCE_KEY.format(context.user_id), ttl)

            if context.session_id is not None:
                # Update score in session presence sorted set
                pipe.zadd(
                    _SESSION_PRESENCE_KEY.format(context.session_id),
                    {context.connection_id: current_epoch},
                )
                pipe.expire(_SESSION_PRESENCE_KEY.format(context.session_id), ttl)

            # Refresh metadata TTL
            pipe.expire(_CONNECTION_METADATA_KEY.format(context.connection_id), ttl)

            await pipe.execute()

    async def get_active_connection_count(self, user_id: str) -> int:
        """Task 3.2: Get connection count using sorted sets (no drift)."""
        if self._redis is None:
            return 0

        user_key = _USER_PRESENCE_KEY.format(user_id)
        current_epoch = time()
        ttl_seconds = self._settings.WEBSOCKET_PRESENCE_TTL_SECONDS
        cutoff_epoch = current_epoch - ttl_seconds

        # Atomic: evict stale + count
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.zremrangebyscore(user_key, "-inf", cutoff_epoch)
            pipe.zcard(user_key)
            results = await pipe.execute()

        return results[1] if len(results) > 1 else 0

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
            raise WebSocketIdleTimeoutError from exc

        # Task 3.1: Check if session is still valid (pull-based revocation)
        await self._check_session_validity(context, websocket)

        # Task 3.3: Direct rate limiter calls (no state mutation)
        await self._apply_rate_limits(context)

        await self.touch_connection(context)
        return payload

    async def send_json(
        self,
        websocket: WebSocket,
        payload: object,
        context: WebSocketSecurityContext,
    ) -> None:
        # Task 3.1: Check session validity before sending
        await self._check_session_validity(context, websocket)

        await websocket.send_json(payload)
        await self.touch_connection(context)

    async def close_with_violation(
        self,
        websocket: WebSocket,
        context: WebSocketSecurityContext,
        violation: WebSocketSecurityViolationError,
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

    async def _check_session_validity(
        self,
        context: WebSocketSecurityContext,
        websocket: WebSocket,  # noqa: ARG002
    ) -> None:
        """Task 3.1: Pull-based revocation check."""
        if self._token_repo is None or context.session_id is None:
            return

        # Re-read session from Redis to check if it's been revoked
        result = await self._token_repo.get_session(context.session_id)
        if isinstance(result, Failure):
            # Log but don't crash - infrastructure error, allow connection to continue
            logger.warning(
                "Failed to check session validity",
                session_id=context.session_id,
                error=str(result.failure()),
            )
            return

        session_data = result.unwrap()
        if session_data is None:
            # Session has been revoked or expired
            logger.info(
                "Session revoked or expired - closing connection",
                session_id=context.session_id,
                user_id=context.user_id,
            )
            raise WebSocketSessionRevokedError

    async def _apply_rate_limits(
        self,
        context: WebSocketSecurityContext,
    ) -> None:
        """Task 3.3: Direct rate limiter calls without state mutation."""
        try:
            # Check user rate limit
            self._user_limiter.try_acquire(context.user_rate_limit_key)
        except BucketFullException as e:
            raise WebSocketRateLimitExceededError from e

        try:
            # Check connection rate limit
            self._connection_limiter.try_acquire(context.connection_rate_limit_key)
        except BucketFullException as e:
            raise WebSocketRateLimitExceededError from e
