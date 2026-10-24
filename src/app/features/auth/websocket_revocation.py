"""Validated, replay-tolerant cross-worker WebSocket revocation events."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from opentelemetry import metrics
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.observability.logging import logger

if TYPE_CHECKING:
    from redis.asyncio import Redis

REVOCATION_STREAM_KEY = "ws:revocation:events"
REVOCATION_EVENT_VERSION = 1
REVOCATION_STREAM_MAXLEN = 10_000
_REVOCATION_METER = metrics.get_meter("app.auth.websocket_revocation")
REVOCATION_PUBLISHED = _REVOCATION_METER.create_counter("websocket.revocation.published")
REVOCATION_PUBLISH_FAILURES = _REVOCATION_METER.create_counter(
    "websocket.revocation.publish_failures"
)
REVOCATION_MALFORMED = _REVOCATION_METER.create_counter("websocket.revocation.malformed")
REVOCATION_RECONNECTS = _REVOCATION_METER.create_counter("websocket.revocation.reconnects")
REVOCATION_CLOSED = _REVOCATION_METER.create_counter("websocket.revocation.closed")
REVOCATION_CLOSE_LATENCY = _REVOCATION_METER.create_histogram(
    "websocket.revocation.close_latency_ms"
)
REVOCATION_SUBSCRIBER_LAG = _REVOCATION_METER.create_histogram(
    "websocket.revocation.subscriber_lag_ms"
)
REVOCATION_STREAM_GAPS = _REVOCATION_METER.create_counter("websocket.revocation.stream_gaps")


class WebSocketRevocationEvent(BaseModel):
    """Internal event contract; Redis is transport, not authorization."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1]
    event_id: str = Field(min_length=16, max_length=128)
    scope: Literal["session", "user"]
    tenant_id: str | None = Field(default=None, min_length=1, max_length=128)
    user_id: str = Field(min_length=1, max_length=256)
    session_id: str | None = Field(default=None, min_length=1, max_length=256)
    except_session_id: str | None = Field(default=None, min_length=1, max_length=256)
    reason: str = Field(min_length=1, max_length=128)
    occurred_at: float = Field(ge=0)
    authz_revision: int | None = Field(default=None, ge=0)
    origin: str = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def validate_scope(self) -> WebSocketRevocationEvent:
        if self.scope == "session" and self.session_id is None:
            msg = "session revocation requires session_id"
            raise ValueError(msg)
        if self.scope == "session" and self.except_session_id is not None:
            msg = "session revocation cannot exclude a session"
            raise ValueError(msg)
        if self.scope == "user" and self.session_id is not None:
            msg = "user revocation cannot target session_id"
            raise ValueError(msg)
        return self


def new_event(
    *,
    scope: Literal["session", "user"],
    user_id: str,
    session_id: str | None = None,
    except_session_id: str | None = None,
    reason: str,
    tenant_id: str | None = None,
    authz_revision: int | None = None,
    origin: str = "auth_repository",
) -> WebSocketRevocationEvent:
    return WebSocketRevocationEvent(
        version=REVOCATION_EVENT_VERSION,
        event_id=uuid4().hex,
        scope=scope,
        tenant_id=tenant_id,
        user_id=user_id,
        session_id=session_id,
        except_session_id=except_session_id,
        reason=reason,
        occurred_at=time.time(),
        authz_revision=authz_revision,
        origin=origin,
    )


async def publish_event(redis: Redis, event: WebSocketRevocationEvent) -> bool:
    """Best-effort durable publication; pull checks remain the security gate."""
    try:
        fields = {
            key: value
            for key, value in event.model_dump(mode="json").items()
            if value is not None
        }
        await redis.xadd(
            REVOCATION_STREAM_KEY,
            fields,
            maxlen=REVOCATION_STREAM_MAXLEN,
            approximate=True,
        )
    except Exception as exc:  # noqa: BLE001 — publication is best effort
        REVOCATION_PUBLISH_FAILURES.add(1)
        logger.bind(
            operation="publish_websocket_revocation",
            event_id=event.event_id,
            user_id=event.user_id,
            error_type=type(exc).__name__,
        ).warning("WebSocket revocation event publication unavailable")
        return False
    logger.bind(
        event_id=event.event_id,
        scope=event.scope,
        user_id=event.user_id,
        session_id=event.session_id,
    ).info("WebSocket revocation event published")
    REVOCATION_PUBLISHED.add(1, {"scope": event.scope})
    return True


def parse_event(fields: object) -> WebSocketRevocationEvent | None:
    if not isinstance(fields, dict):
        return None
    normalized: dict[str, object] = {}
    for key, value in fields.items():
        name = key.decode() if isinstance(key, bytes) else str(key)
        decoded = value.decode() if isinstance(value, bytes) else value
        if name == "version":
            try:
                decoded = int(str(decoded))
            except (TypeError, ValueError):
                return None
        elif name == "occurred_at":
            try:
                decoded = float(str(decoded))
            except (TypeError, ValueError):
                return None
        elif name == "authz_revision" and decoded not in {None, "", b""}:
            try:
                decoded = int(str(decoded))
            except (TypeError, ValueError):
                return None
        normalized[name] = decoded
    try:
        return WebSocketRevocationEvent.model_validate(normalized)
    except (TypeError, ValueError):
        return None
