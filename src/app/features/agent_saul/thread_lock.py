"""Per-thread connection lock (doc §5.5) — Redis ``SET NX`` + TTL with renewal.

One Redis key per thread does three jobs:

- **Mutual exclusion** for connection lifetime: a second connection to the same
  thread is refused with ``THREAD_BUSY`` — two concurrent ``Command(resume=...)``
  calls on one paused checkpoint cannot be recovered cleanly (§5.5).
- **Heartbeat cadence**: renewal is time-driven (every TTL/3), independent of
  WS traffic; a long silent LLM turn must never let the lock lapse mid-run.
- **Aliveness**: a fresh renewal means a live connection; renewal failure
  reports not-alive and the lock stops. Disconnect release is a courtesy —
  **TTL expiry is the fence** and the only correctness-critical crash detection
  (§5.5).

Fail-closed (§9): if Redis is unavailable, acquisition fails and the caller
refuses the connection — never proceed without the exclusion fence.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from redis.exceptions import RedisError

from app.observability.logging import logger

if TYPE_CHECKING:
    from redis.asyncio import Redis

DEFAULT_TTL_SECONDS = 60
_RENEW_INTERVAL = DEFAULT_TTL_SECONDS / 3  # §5.5 [v3]: renewal every TTL/3

_LOCK_KEY_PREFIX = "saul:thread_lock"


def lock_key(thread_id: str) -> str:
    return f"{_LOCK_KEY_PREFIX}:{thread_id}"


@dataclass
class Heartbeat:
    """The aliveness carrier the recorder reads for every projection."""

    fresh: bool = False
    last_renewed_at: float | None = field(default=None, repr=True)


class ThreadLock:
    """Acquire with TTL, periodic renew every TTL/3, best-effort release (§5.5)."""

    def __init__(
        self,
        redis: Redis,
        thread_id: str,
        *,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        renew_interval: float = _RENEW_INTERVAL,
    ) -> None:
        self._redis = redis
        self._thread_id = thread_id
        self._ttl = ttl_seconds
        self._renew_interval = renew_interval
        self._held = False
        self._renew_task: asyncio.Task[None] | None = None
        self.heartbeat = Heartbeat()
        self.log = logger.bind(thread_id=thread_id, component="thread_lock")

    @property
    def is_held(self) -> bool:
        return self._held

    def is_alive(self) -> bool:
        """Aliveness input for projections (§4.1): a fresh heartbeat only."""
        return self._held and self.heartbeat.fresh

    async def acquire(self) -> bool:
        """``SET NX`` — True when acquired; False means ``THREAD_BUSY`` or outage."""
        try:
            acquired = await self._redis.set(lock_key(self._thread_id), "1", nx=True, ex=self._ttl)
        except RedisError:
            self.log.warning("thread_lock_redis_unavailable")
            self._held = False
            self.heartbeat.fresh = False
            return False
        self._held = bool(acquired)
        if self._held:
            self.heartbeat.fresh = True
            self._start_renew_loop()
        return self._held

    async def release(self) -> None:
        """Best-effort; TTL expiry remains the correctness fence (§5.5)."""
        self._held = False
        self.heartbeat.fresh = False
        renew_task = self._renew_task
        self._renew_task = None
        if renew_task is not None and not renew_task.done():
            renew_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await renew_task
        try:
            await self._redis.delete(lock_key(self._thread_id))
        except RedisError:
            self.log.warning("thread_lock_release_lapsed")

    def _stop_renew_loop(self) -> None:
        renew_task = self._renew_task
        if renew_task is not None and not renew_task.done():
            renew_task.cancel()
        self._renew_task = None

    def _start_renew_loop(self) -> None:
        if self._renew_task is not None and not self._renew_task.done():
            return
        self._renew_task = asyncio.create_task(
            self._renew_loop(), name=f"saul-thread-lock-renew-{self._thread_id}"
        )

    async def _renew_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._renew_interval)
                if not self._held:
                    return
                try:
                    renewed = await self._renew_once()
                except RedisError:
                    self.log.warning("thread_lock_renew_failed")
                    self.heartbeat.fresh = False
                    self._held = False  # stop renewing; TTL fence absorbs the rest
                    return
                if renewed:
                    self.heartbeat.fresh = True
                else:
                    self.log.warning("thread_lock_lost")
                    self.heartbeat.fresh = False
                    self._held = False
                    return
        except asyncio.CancelledError:
            # Best-effort release on cancellation; absence of the key is then
            # the TTL's job, not ours (§5.5 disconnect release is a courtesy).
            if self._held:
                with contextlib.suppress(RedisError):
                    await self._redis.delete(lock_key(self._thread_id))
            raise

    async def _renew_once(self) -> bool:
        """Single renewal with an atomic check-and-expire; False when lost."""
        key = lock_key(self._thread_id)
        renewed = await self._redis.expire(key, int(self._ttl))
        return bool(renewed)
