from __future__ import annotations

import asyncio
import inspect
import json
from contextlib import suppress
from datetime import UTC, datetime
from time import monotonic, time
from typing import (
    TYPE_CHECKING,
    cast,
)

from fastapi import WebSocket, WebSocketException, status
from pydantic import BaseModel, ConfigDict, Field
from pyrate_limiter import BucketFullException, Limiter, Rate
from pyrate_limiter.buckets import InMemoryBucket, RedisBucket
from returns.result import Failure

from app.features.auth.websocket_revocation import (
    REVOCATION_CLOSE_LATENCY,
    REVOCATION_CLOSED,
    REVOCATION_CURSOR_MARKER_FIELD,
    REVOCATION_MALFORMED,
    REVOCATION_RECONNECTS,
    REVOCATION_STREAM_GAPS,
    REVOCATION_STREAM_KEY,
    REVOCATION_STREAM_MAXLEN,
    REVOCATION_SUBSCRIBER_LAG,
    parse_event,
)
from app.observability.logging import logger
from app.security.tokens import TokenClaims

if TYPE_CHECKING:
    from redis.asyncio import Redis

    from app.config import Settings
    from app.features.auth.repository import RefreshTokenRepository
    from app.features.auth.websocket_revocation import WebSocketRevocationEvent

# Task 3.2: New sorted set key patterns
_USER_PRESENCE_KEY = "ws:user:{}"  # Sorted set: member=connection_id, score=last_touch_epoch
_SESSION_PRESENCE_KEY = "ws:session:{}"  # Sorted set: member=connection_id, score=last_touch_epoch
_CONNECTION_METADATA_KEY = "ws:connection:{}"  # Hash: connection metadata
_REVOCATION_REVISION_TTL_SECONDS = 3600.0
_REVOCATION_CLOSE_CONCURRENCY = 20
_REVOCATION_CLOSE_TIMEOUT_SECONDS = 1.0


class WebSocketSecurityContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    claims: TokenClaims
    user_id: str
    session_id: str | None
    connection_id: str
    origin: str | None
    user_rate_limit_key: str
    connection_rate_limit_key: str
    tenant_id: str | None = None
    connected_at: float = Field(default_factory=time, ge=0)


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


# `claims: TokenClaims` was historically a TYPE_CHECKING-only import; resolve the
# forward reference so WebSocketSecurityContext can validate at runtime.
WebSocketSecurityContext.model_rebuild(_types_namespace={"TokenClaims": TokenClaims})


def _raise_websocket_rate_limit(*_: object, **__: object) -> None:
    raise WebSocketRateLimitExceededError


async def build_websocket_security_service(
    redis: Redis | None,
    settings: Settings,
    token_repo: RefreshTokenRepository | None = None,
) -> WebSocketSecurityService:
    # pyrate v3's Rate takes milliseconds; settings are in seconds.
    user_rates = [
        Rate(
            settings.WEBSOCKET_USER_MESSAGE_RATE,
            settings.WEBSOCKET_USER_MESSAGE_PERIOD_SECONDS * 1000,
        )
    ]
    connection_rates = [
        Rate(
            settings.WEBSOCKET_CONNECTION_MESSAGE_RATE,
            settings.WEBSOCKET_CONNECTION_MESSAGE_PERIOD_SECONDS * 1000,
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
            logger.bind(
                operation="build_websocket_security", layer="websocket", error=str(exc)
            ).warning("Redis-based WebSocket rate limiter failed, falling back to in-memory bucket")
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
        self._presence_score: dict[str, float] = {}
        # Outbound token streams can emit hundreds of frames per turn. Track
        # the last completed session check with a monotonic clock so revocation
        # stays bounded without serializing every token behind Redis.
        self._last_outbound_session_check: dict[str, float] = {}
        self._outbound_session_check_locks: dict[str, asyncio.Lock] = {}
        # Task 3.4: live connection registry so a connection id can be closed
        # without holding the WebSocket object at the call site.
        self._live_connections: dict[str, tuple[WebSocket, WebSocketSecurityContext]] = {}
        # New workers tail from now. Replaying retained events could revoke a
        # socket created after the revocation; reconnects resume their cursor.
        self._revocation_last_id = "$"
        self._revocation_cursor_initialized = False
        self._revocation_cursor_ready = asyncio.Event()
        self._revocation_activity = asyncio.Event()
        self._seen_revocation_events: dict[str, None] = {}
        self._latest_revocation_revisions: dict[str, tuple[int, float]] = {}
        self._max_seen_revocation_events = 4096
        self._max_revocation_revisions = 4096
        self._revocation_close_semaphore = asyncio.Semaphore(_REVOCATION_CLOSE_CONCURRENCY)
        self._revocation_needs_gap_check = False

    def close(self) -> None:
        """Dispose rate-limit buckets and stop pyrate's background leak tasks."""
        for limiter in (self._user_limiter, self._connection_limiter):
            for bucket in limiter.buckets():
                limiter.dispose(bucket)

    @property
    def redis(self) -> Redis | None:
        """Expose the Redis client (satisfies AuthService's WebSocketConnectionCloser)."""
        return self._redis

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
        *, claims: TokenClaims, origin: str | None, connection_id: str, tenant_id: str | None = None
    ) -> WebSocketSecurityContext:
        return WebSocketSecurityContext(
            claims=claims,
            user_id=claims.sub,
            session_id=claims.sid,
            connection_id=connection_id,
            origin=origin,
            user_rate_limit_key=f"user:{claims.sub}",
            connection_rate_limit_key=f"connection:{connection_id}",
            tenant_id=tenant_id,
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

    async def register_connection(
        self,
        context: WebSocketSecurityContext,
        websocket: WebSocket | None = None,
    ) -> None:
        """Task 3.2/3.4: Register connection with an atomic capacity gate."""
        if self._redis is None:
            return

        current_epoch = time()
        ttl = self._settings.WEBSOCKET_PRESENCE_TTL_SECONDS
        user_key = _USER_PRESENCE_KEY.format(context.user_id)

        # Atomic check-and-increment: eviction, add, and count execute as one
        # Redis transaction, so concurrent registrations are serialized and the
        # count each transaction observes includes all prior ones. If we
        # overflow, roll back our own entry — final count never exceeds max.
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.zremrangebyscore(user_key, "-inf", current_epoch - ttl)
            pipe.zadd(user_key, {context.connection_id: current_epoch})
            pipe.zcard(user_key)
            results = await pipe.execute()

        active_count = int(results[2]) if len(results) > 2 else 0
        if active_count > self._settings.WEBSOCKET_MAX_CONNECTIONS_PER_USER:
            await self._redis.zrem(user_key, context.connection_id)
            raise WebSocketException(
                code=status.WS_1008_POLICY_VIOLATION,
                reason="Maximum concurrent WebSocket connections exceeded",
            )

        metadata = json.dumps(
            {
                "user_id": context.user_id,
                "session_id": context.session_id,
                "created_at": datetime.now(UTC).isoformat(),
            }
        )

        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.expire(user_key, ttl)

            # Add to session presence sorted set (for revocation lookup)
            if context.session_id is not None:
                session_key = _SESSION_PRESENCE_KEY.format(context.session_id)
                pipe.zadd(session_key, {context.connection_id: current_epoch})
                pipe.expire(session_key, ttl)

            # Store connection metadata
            pipe.setex(_CONNECTION_METADATA_KEY.format(context.connection_id), ttl, metadata)

            await pipe.execute()

        # Initialize last touch time for throttling
        self._last_touch_time[context.connection_id] = current_epoch
        self._presence_score[context.connection_id] = current_epoch

        if websocket is not None:
            self._live_connections[context.connection_id] = (websocket, context)

    async def unregister_connection(self, context: WebSocketSecurityContext) -> None:
        """Task 3.2: Unregister connection from sorted sets."""
        # In-process state must be cleared even when Redis is absent or the
        # distributed cleanup below fails.
        self._last_touch_time.pop(context.connection_id, None)
        self._presence_score.pop(context.connection_id, None)
        self._last_outbound_session_check.pop(context.connection_id, None)
        self._outbound_session_check_locks.pop(context.connection_id, None)
        self._live_connections.pop(context.connection_id, None)
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

    async def touch_connection(self, context: WebSocketSecurityContext) -> None:
        """Task 3.2: Update connection presence with throttling (Finding 1)."""
        if self._redis is None:
            return

        current_epoch = time()
        # Throttle: only touch if 30 seconds have elapsed since last touch
        last_touch = self._last_touch_time.get(context.connection_id, 0)
        if current_epoch - last_touch < 30:
            return

        last_score = self._presence_score.get(context.connection_id, last_touch)
        current_epoch = max(current_epoch, last_score + 0.000001)

        self._last_touch_time[context.connection_id] = current_epoch
        self._presence_score[context.connection_id] = current_epoch
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
        # Outbound streams retain a bounded revocation check without putting a
        # Redis round trip on every token frame. Inbound frames remain checked
        # unconditionally in receive_json().
        await self._check_outbound_session_validity_if_due(context, websocket)

        await websocket.send_json(payload)
        await self.touch_connection(context)

    async def close_with_violation(
        self,
        websocket: WebSocket,
        violation: WebSocketSecurityViolationError,
    ) -> None:
        with suppress(Exception):
            await websocket.send_json(self._violation_frame(violation))
        with suppress(Exception):
            await websocket.close(code=violation.close_code, reason=violation.message)

    @staticmethod
    def _violation_frame(violation: WebSocketSecurityViolationError) -> dict[str, object]:
        return {
            "type": "error",
            "node": None,
            "code": violation.error_code,
            "message": violation.message,
            "retryable": violation.retryable,
        }

    async def close_connection(self, connection_id: str, /, *, reason: str) -> None:
        """Task 3.4: Close a live connection by id.

        No-op when the connection is not registered in this process (e.g. it
        belongs to another worker or already disconnected).
        """
        entry = self._live_connections.pop(connection_id, None)
        if entry is None:
            return

        websocket, context = entry
        safe_reason = self._safe_close_reason(reason)
        violation = WebSocketSecurityViolationError(
            error_code="SESSION_REVOKED",
            message=f"Connection closed: {safe_reason}",
            retryable=False,
        )
        try:
            with suppress(Exception):
                await websocket.send_json(self._violation_frame(violation))
            with suppress(Exception):
                await websocket.close(code=violation.close_code, reason=safe_reason)
        except asyncio.CancelledError:
            # The listener applies a per-socket deadline. Cancellation while a
            # client is stuck in send_json must still get a best-effort close;
            # otherwise the registry removal below makes that socket
            # impossible for later sweeps to find.
            with suppress(BaseException):
                await asyncio.shield(websocket.close(code=violation.close_code, reason=safe_reason))
            raise
        finally:
            with suppress(Exception):
                await self.unregister_connection(context)

    @staticmethod
    def _safe_close_reason(reason: str) -> str:
        """Keep untrusted stream text valid and bounded for WebSocket close."""
        cleaned = "".join(" " if ord(char) < 32 or ord(char) == 127 else char for char in reason)
        encoded = cleaned.encode("utf-8")[:123]
        while True:
            try:
                return encoded.decode("utf-8")
            except UnicodeDecodeError:
                encoded = encoded[:-1]

    async def run_revocation_loop(self, interval_seconds: float = 30.0) -> None:
        """Task 3.1: Periodic pull-based revocation sweep over live connections.

        Every ``interval_seconds`` each live connection's session is re-read
        from Redis; connections whose session is gone are closed with a
        SESSION_REVOKED violation. Bounded staleness: one interval.
        """
        while True:
            await asyncio.sleep(interval_seconds)
            try:
                await self._sweep_revoked_connections()
            except Exception as exc:  # noqa: BLE001 — monitor must survive any failure
                logger.bind(operation="revocation_loop", layer="websocket", error=str(exc)).warning(
                    "Revocation sweep failed"
                )

    async def run_revocation_listener(self) -> None:
        """Consume durable cross-worker revocations with pull-based fallback.

        Streams are intentionally consumed without consumer groups: every
        worker needs the event, and an in-memory cursor avoids pending-entry
        ownership during reconnects.  Trimming or outages never weaken
        security because the periodic and outbound session checks remain.
        """
        if self._redis is None or not getattr(self._settings, "WEBSOCKET_REVOCATION_ENABLED", True):
            return
        base_delay = max(
            0.05,
            float(getattr(self._settings, "WEBSOCKET_REVOCATION_RECONNECT_BASE_SECONDS", 0.25)),
        )
        max_delay = max(
            base_delay,
            float(getattr(self._settings, "WEBSOCKET_REVOCATION_RECONNECT_MAX_SECONDS", 5.0)),
        )
        batch_size = max(1, int(getattr(self._settings, "WEBSOCKET_REVOCATION_BATCH_SIZE", 100)))
        delay = base_delay
        while True:
            try:
                if not self._revocation_cursor_initialized:
                    tail = await self._redis.xrevrange(REVOCATION_STREAM_KEY, count=1)
                    if tail:
                        self._revocation_last_id = self._decode_stream_id(tail[0][0])
                    else:
                        # An empty stream has no concrete "future" ID.  A
                        # private marker establishes one so an outage between
                        # the initial read and the first event cannot be
                        # hidden by a subsequent XREAD from ``$``.
                        marker_id = await self._redis.xadd(
                            REVOCATION_STREAM_KEY,
                            {REVOCATION_CURSOR_MARKER_FIELD: "1"},
                            maxlen=REVOCATION_STREAM_MAXLEN,
                            approximate=True,
                        )
                        self._revocation_last_id = self._decode_stream_id(marker_id)
                    self._revocation_cursor_initialized = True
                    self._revocation_cursor_ready.set()
                records = await self._redis.xread(
                    {REVOCATION_STREAM_KEY: self._revocation_last_id},
                    # Keep the blocking interval below the default Redis
                    # socket timeout so idle listeners reconnect cleanly.
                    block=1000,
                    count=batch_size,
                )
                delay = base_delay
                if self._revocation_needs_gap_check:
                    self._revocation_needs_gap_check = not await self._check_revocation_stream_gap()
                for _stream, entries in records or []:
                    stream_entries = cast("list[tuple[object, object]]", entries)
                    for stream_id, fields in stream_entries:
                        self._revocation_last_id = self._decode_stream_id(stream_id)
                        if isinstance(fields, dict) and any(
                            (key.decode() if isinstance(key, bytes) else str(key))
                            == REVOCATION_CURSOR_MARKER_FIELD
                            for key in fields
                        ):
                            self._revocation_activity.set()
                            continue
                        event = parse_event(fields)
                        if event is None:
                            logger.warning("Ignoring malformed WebSocket revocation event")
                            REVOCATION_MALFORMED.add(1)
                            self._revocation_activity.set()
                            continue
                        subscriber_lag_ms = max(0.0, (time() - event.occurred_at) * 1000)
                        REVOCATION_SUBSCRIBER_LAG.record(
                            subscriber_lag_ms,
                            {"scope": event.scope},
                        )
                        await self._handle_revocation_event(event)
                        self._revocation_activity.set()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — listener must reconnect
                REVOCATION_RECONNECTS.add(1)
                self._revocation_needs_gap_check = True
                logger.bind(operation="websocket_revocation_listener", error=str(exc)).warning(
                    "WebSocket revocation stream unavailable; pull checks remain active"
                )
                await asyncio.sleep(delay)
                delay = min(max_delay, delay * 2)

    @staticmethod
    def _decode_stream_id(value: object) -> str:
        return value.decode() if isinstance(value, bytes) else str(value)

    async def _check_revocation_stream_gap(self) -> bool:
        if self._redis is None or self._revocation_last_id == "$":
            return True
        try:
            info = await self._redis.xinfo_stream(REVOCATION_STREAM_KEY)
            first_entry = info.get("first-entry") or info.get(b"first-entry")
            if not first_entry:
                return True
            first_id = self._decode_stream_id(first_entry[0])
            if self._stream_id_tuple(self._revocation_last_id) < self._stream_id_tuple(first_id):
                REVOCATION_STREAM_GAPS.add(1)
                logger.bind(
                    operation="websocket_revocation_stream_gap",
                    cursor=self._revocation_last_id,
                    first_available=first_id,
                ).warning("WebSocket revocation stream entries were trimmed during outage")
            else:
                return True
            return True  # noqa: TRY300
        except Exception as exc:  # noqa: BLE001 — diagnostics cannot stop reconnect
            logger.bind(operation="websocket_revocation_stream_gap_check", error=str(exc)).warning(
                "Unable to check WebSocket revocation stream continuity"
            )
            return False

    @staticmethod
    def _stream_id_tuple(value: str) -> tuple[int, int]:
        major, minor = value.split("-", maxsplit=1)
        return int(major), int(minor)

    async def _handle_revocation_event(self, event: WebSocketRevocationEvent) -> None:  # noqa: PLR0912
        """Apply one validated event only to matching local sockets."""
        if event.event_id in self._seen_revocation_events:
            return
        now_monotonic = monotonic()
        expired_users = [
            user_id
            for user_id, (_revision, expires_at) in self._latest_revocation_revisions.items()
            if expires_at <= now_monotonic
        ]
        for user_id in expired_users:
            self._latest_revocation_revisions.pop(user_id, None)
        while len(self._latest_revocation_revisions) >= self._max_revocation_revisions:
            self._latest_revocation_revisions.pop(next(iter(self._latest_revocation_revisions)))
        if event.authz_revision is not None:
            previous_revision_entry = self._latest_revocation_revisions.get(event.user_id)
            if previous_revision_entry is not None and now_monotonic >= previous_revision_entry[1]:
                self._latest_revocation_revisions.pop(event.user_id, None)
                previous_revision_entry = None
            previous_revision = (
                previous_revision_entry[0] if previous_revision_entry is not None else None
            )
            if previous_revision is not None and event.authz_revision < previous_revision:
                return
            self._latest_revocation_revisions[event.user_id] = (
                event.authz_revision,
                now_monotonic + _REVOCATION_REVISION_TTL_SECONDS,
            )
        self._seen_revocation_events[event.event_id] = None
        if len(self._seen_revocation_events) > self._max_seen_revocation_events:
            self._seen_revocation_events.pop(next(iter(self._seen_revocation_events)))

        matching: list[str] = []
        for connection_id, (_websocket, context) in list(self._live_connections.items()):
            if event.user_id != context.user_id or event.tenant_id != context.tenant_id:
                continue
            if event.occurred_at <= context.connected_at:
                continue
            event_revision = event.authz_revision
            context_revision = context.claims.authz_revision
            if event_revision is not None and context_revision is not None and context_revision > event_revision:
                continue
            if event.scope == "session":
                if event.session_id == context.session_id:
                    matching.append(connection_id)
            elif (
                context.session_id != event.except_session_id
                and (event.session_ids is None or context.session_id in event.session_ids)
            ):
                matching.append(connection_id)

        async def close_one(connection_id: str) -> None:
            async with self._revocation_close_semaphore:
                try:
                    await asyncio.wait_for(
                        self.close_connection(connection_id, reason=event.reason),
                        timeout=_REVOCATION_CLOSE_TIMEOUT_SECONDS,
                    )
                except TimeoutError:
                    logger.bind(
                        operation="websocket_revocation_close",
                        event_id=event.event_id,
                        connection_id=connection_id,
                    ).warning("Timed out closing revoked WebSocket connection")
                except Exception as exc:  # noqa: BLE001 — continue other closes
                    logger.bind(
                        operation="websocket_revocation_close",
                        event_id=event.event_id,
                        connection_id=connection_id,
                        error_type=type(exc).__name__,
                    ).warning("Failed to close revoked WebSocket connection")

        await asyncio.gather(*(close_one(connection_id) for connection_id in matching))
        if matching:
            REVOCATION_CLOSED.add(len(matching), {"scope": event.scope})
            close_latency_ms = max(0.0, (time() - event.occurred_at) * 1000)
            REVOCATION_CLOSE_LATENCY.record(close_latency_ms, {"scope": event.scope})
            logger.bind(
                operation="websocket_revocation_close",
                event_id=event.event_id,
                origin=event.origin,
                scope=event.scope,
                close_latency_ms=round(close_latency_ms, 2),
            ).info("WebSocket revocation applied locally")

    async def _sweep_revoked_connections(self) -> None:
        if self._token_repo is None:
            return

        for connection_id, (_websocket, context) in list(self._live_connections.items()):
            if context.session_id is None:
                continue

            result = await self._token_repo.get_session(context.session_id)
            if isinstance(result, Failure):
                # Fail open during the periodic sweep only — the sweep is a
                # monitor, not a gate; the per-message check (
                # _check_session_validity) is fail-closed on infrastructure
                # errors, so an outage still kills connections on next frame.
                logger.bind(
                    operation="sweep_revoked_connections",
                    layer="websocket",
                    session_id=context.session_id,
                    error=str(result.failure()),
                ).warning("Failed to check session validity")
                continue

            session = result.unwrap()
            if session is None or session.expires_at < datetime.now(UTC):
                logger.info(
                    "Session revoked or expired - closing connection",
                    session_id=context.session_id,
                    user_id=context.user_id,
                )
                await self.close_connection(connection_id, reason="session_revoked")

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
            # Fail-closed on outage: if the session cannot be verified, the
            # connection dies. Matches agent_saul's resolve_ws_session, which
            # rejects the handshake on UNAVAILABLE — never fail open.
            logger.bind(
                operation="check_session_validity",
                layer="websocket",
                event="session_check_unavailable",
                session_id=context.session_id,
                user_id=context.user_id,
                error=str(result.failure()),
            ).warning("Session check unavailable - closing connection (fail-closed)")
            raise WebSocketSessionRevokedError

        session_data = result.unwrap()
        if session_data is None:
            # Session has been revoked or expired
            logger.info(
                "Session revoked or expired - closing connection",
                session_id=context.session_id,
                user_id=context.user_id,
            )
            raise WebSocketSessionRevokedError
        if session_data.user_id != context.user_id:
            # Owner mismatch: the session row belongs to another user. The wire
            # code stays SESSION_REVOKED; the distinct log event is the
            # forensic signal (defense-in-depth, fail-closed).
            logger.bind(
                operation="check_session_validity",
                layer="websocket",
                event="session_owner_mismatch",
                session_id=context.session_id,
                user_id=context.user_id,
                owner_user_id=session_data.user_id,
            ).warning("Session owner mismatch - closing connection")
            raise WebSocketSessionRevokedError
        # ponytail: expires_at is authoritative; Redis TTL may lag or be 0
        if session_data.expires_at < datetime.now(UTC):
            logger.info(
                "Session expired (expires_at in past) - closing connection",
                session_id=context.session_id,
                user_id=context.user_id,
            )
            raise WebSocketSessionRevokedError

    async def _check_outbound_session_validity_if_due(
        self,
        context: WebSocketSecurityContext,
        websocket: WebSocket,
    ) -> None:
        """Check outbound-session validity at most once per configured interval.

        A per-connection lock makes the refresh single-flight: concurrent sends
        cannot bypass an in-flight check, while followers reuse its result.
        The timestamp is stored only after a successful validity check.
        Infrastructure failures, revoked sessions, owner mismatches, and
        expired sessions all raise and never populate or refresh the cache.
        """
        if self._token_repo is None or context.session_id is None:
            return
        check_lock = self._outbound_session_check_locks.get(context.connection_id)
        if check_lock is None:
            # No await between lookup and store: creation is atomic with
            # respect to other tasks on this event loop.
            check_lock = asyncio.Lock()
            self._outbound_session_check_locks[context.connection_id] = check_lock
        async with check_lock:
            attempted_at = monotonic()
            last_attempt = self._last_outbound_session_check.get(context.connection_id)
            interval = self._settings.WEBSOCKET_OUTBOUND_SESSION_CHECK_INTERVAL_SECONDS
            if last_attempt is not None and attempted_at - last_attempt < interval:
                return
            await self._check_session_validity(context, websocket)
            self._last_outbound_session_check[context.connection_id] = monotonic()

    @staticmethod
    async def _acquire_rate(limiter: Limiter, key: str) -> None:
        # pyrate v3 semantics: with the default raise_when_fail=True a full
        # bucket raises BucketFullException (mapped below); otherwise
        # try_acquire returns False. It also returns an Awaitable when the
        # bucket is async-backed (RedisBucket) — not awaiting it silently
        # disables the rate limit entirely.
        try:
            result = limiter.try_acquire(key)
            if inspect.isawaitable(result):
                result = await result
        except BucketFullException as e:
            raise WebSocketRateLimitExceededError from e
        if result is False:
            raise WebSocketRateLimitExceededError

    async def _apply_rate_limits(
        self,
        context: WebSocketSecurityContext,
    ) -> None:
        """Task 3.3: Direct rate limiter calls without state mutation."""
        # Check user rate limit, then connection rate limit
        await self._acquire_rate(self._user_limiter, context.user_rate_limit_key)
        await self._acquire_rate(self._connection_limiter, context.connection_rate_limit_key)
