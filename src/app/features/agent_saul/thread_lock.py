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

Ownership (§5.5): every holder writes a random token as the key value, and
renew/release act on the key only when the stored value still equals that
token (`WATCH`/`MULTI` compare-and-act). An expired holder can therefore never
delete or extend a successor's lease.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable
from dataclasses import dataclass, field
from traceback import format_exc
from typing import TYPE_CHECKING
from uuid import uuid4

from redis.exceptions import RedisError

from app.observability.logging import logger

if TYPE_CHECKING:
    from redis.asyncio import Redis

DEFAULT_TTL_SECONDS = 60

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
        renew_interval: float | None = None,
    ) -> None:
        if ttl_seconds <= 0:
            msg = "ttl_seconds must be positive"
            raise ValueError(msg)
        resolved_renew_interval = ttl_seconds / 3 if renew_interval is None else renew_interval
        if resolved_renew_interval <= 0:
            msg = "renew_interval must be positive"
            raise ValueError(msg)
        self._redis = redis
        self._thread_id = thread_id
        self._ttl = ttl_seconds
        self._renew_interval = resolved_renew_interval
        self._held = False
        self._token: str | None = None
        self._renew_task: asyncio.Task[None] | None = None
        self.heartbeat = Heartbeat()
        self.lease_lost = asyncio.Event()
        self.log = logger.bind(thread_id=thread_id, component="thread_lock")

    @property
    def is_held(self) -> bool:
        return self._held

    def is_alive(self) -> bool:
        """Aliveness input for projections (§4.1): a fresh heartbeat only.

        False once the lease is lost or Redis is unavailable — the projector
        must conclude nothing and the service must stop its run (§5.3, §6).
        """
        return self._held and self.heartbeat.fresh

    async def acquire(self) -> bool:
        """``SET NX`` with a random token — True when acquired; False means
        ``THREAD_BUSY`` or outage."""
        token = uuid4().hex
        try:
            acquired = await self._redis.set(
                lock_key(self._thread_id), token, nx=True, ex=self._ttl
            )
        except RedisError as exc:
            self._log_redis_failure(exc, operation="acquire", event="thread_lock_redis_unavailable")
            self._held = False
            self.heartbeat.fresh = False
            return False
        self._held = bool(acquired)
        if self._held:
            self._token = token
            self.heartbeat.fresh = True
            self._start_renew_loop()
        return self._held

    async def release(self) -> None:
        """Best-effort owner-checked delete; TTL expiry remains the fence (§5.5)."""
        self._held = False
        self.heartbeat.fresh = False
        token = self._token
        self._token = None
        renew_task = self._renew_task
        self._renew_task = None
        if renew_task is not None and not renew_task.done():
            renew_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await renew_task
        try:
            await self._compare_and_act("release", delete=True, token=token)
        except RedisError as exc:
            self._log_redis_failure(exc, operation="release", event="thread_lock_release_lapsed")

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
                except RedisError as exc:
                    self._log_redis_failure(
                        exc,
                        operation="renew",
                        event="thread_lock_renew_failed",
                    )
                    self.heartbeat.fresh = False
                    self._held = False  # lease lost: stop renewing, stop aliveness
                    self.lease_lost.set()  # unknowable fence → the run must stop
                    return
                if renewed:
                    self.heartbeat.fresh = True
                else:
                    self.log.warning("thread_lock_lost")
                    self.heartbeat.fresh = False
                    self._held = False
                    self.lease_lost.set()  # the RUN sees this even mid-stream
                    return
        except asyncio.CancelledError:
            # Best-effort owner-checked release on cancellation; when the
            # token no longer matches, the key belongs to someone else.
            if self._held:
                token, self._token, self._held = self._token, None, False
                with contextlib.suppress(RedisError):
                    await self._compare_and_act("release", delete=True, token=token)
            raise

    async def _renew_once(self) -> bool:
        """Single owner-checked expiry extension; False when the lease is gone."""
        return await self._compare_and_act("renew", delete=False)

    async def _compare_and_act(self, stage: str, *, delete: bool, token: str | None = None) -> bool:
        """WATCH/MULTI compare-and-act: act only while the token still owns.

        ``delete`` selects owner-checked DEL vs PEXPIRE. Runs atomically from
        the WATCH point; a concurrent steal repoints or removes the key first
        and our branch becomes a no-op returning False.
        """
        key = lock_key(self._thread_id)
        owner = self._token if token is None else token
        if owner is None:
            return False
        async with self._redis.pipeline() as pipe:
            await pipe.watch(key)
            stored = await pipe.get(key)
            # The SDK pool (redis_fastapi) builds without decode_responses, so
            # the same key can read back as bytes on this path and str
            # elsewhere; compare as text (P5a — genuinely dynamic bytes|str).
            stored_text = stored.decode() if isinstance(stored, bytes) else stored
            if stored_text != owner:
                # redis-py asyncio defines unwatch() as a coroutine while
                # fakeredis's is sync — await only a real awaitable (P5a).
                unwatched = pipe.unwatch()
                if isinstance(unwatched, Awaitable):
                    await unwatched
                self.log.bind(stage=stage).info("thread_lock_not_owner")
                return False
            pipe.multi()  # transaction marker — sync command in redis-py pipelines
            if delete:
                pipe.delete(key)
            else:
                pipe.pexpire(key, self._ttl * 1000)
            await pipe.execute()
            return True

    def _log_redis_failure(self, exc: RedisError, *, operation: str, event: str) -> None:
        """Preserve the discarded Redis exception's internal diagnostics."""
        exc.add_note(f"thread_id={self._thread_id}, operation={operation}")
        exc.add_note(format_exc())
        self.log.bind(
            operation=operation,
            error_notes=list(getattr(exc, "__notes__", ())),
        ).warning(event)
