"""
Idempotency primitives for agent tool execution.

`ToolResult` is the normalized envelope returned by tools. `IdempotencyGuard`
provides a two-layer cache backed by Redis for the hot path and Postgres for
durable audit/history.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import text

from app.utils import logger

if TYPE_CHECKING:
    from typing import Any

    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import AsyncEngine

_REDIS_TTL_SECONDS = 86_400
_POSTGRES_TTL_DAYS = 30
_REDIS_KEY_PREFIX = "idempotency:"


class ToolResult(BaseModel):
    """Normalized tool output envelope."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    success: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def ok(cls, data: dict[str, Any], **meta: Any) -> ToolResult:
        return cls(success=True, data=data, metadata=meta)

    @classmethod
    def fail(cls, error: str, **meta: Any) -> ToolResult:
        return cls(success=False, data={}, error=error, metadata=meta)


class IdempotencyGuard:
    """Redis-first, Postgres-backed idempotency guard for tool calls."""

    def __init__(self, redis: Redis, db_engine: AsyncEngine) -> None:
        self._redis = redis
        self._db_engine = db_engine
        self._log = logger.bind(component="idempotency_guard")

    @staticmethod
    def make_key(
        step_id: str,
        input_data: dict[str, Any],
        user_id: str,
    ) -> str:
        """Build a deterministic SHA-256 key for a tool invocation."""
        payload = json.dumps(
            {"step_id": step_id, "input": input_data, "user_id": user_id},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    async def get(self, key: str) -> ToolResult | None:
        """Return a cached tool result when a prior execution exists."""
        redis_value = await self._redis.get(_redis_key(key))
        if redis_value:
            self._log.debug("idempotency_hit_redis", key_prefix=key[:16])
            return ToolResult.model_validate_json(redis_value)

        postgres_result = await self._get_from_postgres(key)
        if postgres_result is None:
            return None

        await self._warm_redis_cache(key=key, result=postgres_result)
        self._log.debug("idempotency_hit_postgres", key_prefix=key[:16])
        return postgres_result

    async def set(
        self,
        key: str,
        result: ToolResult,
        tool_name: str,
        user_id: str,
        thread_id: str,
        step_id: str,
    ) -> None:
        """Persist an execution result to Redis and Postgres."""
        result_json = result.model_dump_json()
        expires_at = datetime.now(tz=UTC) + timedelta(days=_POSTGRES_TTL_DAYS)

        try:
            await self._redis.set(
                _redis_key(key),
                result_json,
                ex=_REDIS_TTL_SECONDS,
            )
        except Exception as exc:
            self._log.bind(error=str(exc), tool_name=tool_name).warning(
                "Idempotency Redis write failed; continuing with Postgres."
            )

        await self._set_in_postgres(
            key=key,
            result_json=result_json,
            tool_name=tool_name,
            user_id=user_id,
            thread_id=thread_id,
            step_id=step_id,
            expires_at=expires_at,
        )
        self._log.bind(tool_name=tool_name, key_prefix=key[:16]).debug(
            "Idempotency state written."
        )

    async def _warm_redis_cache(self, key: str, result: ToolResult) -> None:
        try:
            await self._redis.set(
                _redis_key(key),
                result.model_dump_json(),
                ex=_REDIS_TTL_SECONDS,
            )
        except Exception as exc:
            self._log.bind(error=str(exc), key_prefix=key[:16]).warning(
                "Idempotency Redis cache warm failed."
            )

    async def _get_from_postgres(self, key: str) -> ToolResult | None:
        query = text(
            """
            SELECT result
            FROM tool_executions
            WHERE idempotency_key = :key
              AND (expires_at IS NULL OR expires_at > NOW())
            LIMIT 1
            """
        )
        try:
            async with self._db_engine.connect() as connection:
                row = (await connection.execute(query, {"key": key})).fetchone()
        except Exception as exc:
            self._log.bind(error=str(exc), key_prefix=key[:16]).warning(
                "Idempotency Postgres read failed."
            )
            return None

        if row is None:
            return None
        return ToolResult.model_validate(row[0])

    async def _set_in_postgres(
        self,
        key: str,
        result_json: str,
        tool_name: str,
        user_id: str,
        thread_id: str,
        step_id: str,
        expires_at: datetime,
    ) -> None:
        query = text(
            """
            INSERT INTO tool_executions
                (idempotency_key, tool_name, user_id, thread_id, step_id, result, expires_at)
            VALUES
                (:key, :tool_name, :user_id, :thread_id, :step_id, CAST(:result AS JSONB), :expires_at)
            ON CONFLICT (idempotency_key) DO NOTHING
            """
        )
        try:
            async with self._db_engine.begin() as connection:
                await connection.execute(
                    query,
                    {
                        "key": key,
                        "tool_name": tool_name,
                        "user_id": user_id,
                        "thread_id": thread_id,
                        "step_id": step_id,
                        "result": result_json,
                        "expires_at": expires_at,
                    },
                )
        except Exception as exc:
            self._log.bind(
                error=str(exc),
                key_prefix=key[:16],
                tool_name=tool_name,
            ).error("Idempotency Postgres write failed.")


def _redis_key(key: str) -> str:
    return f"{_REDIS_KEY_PREFIX}{key}"
