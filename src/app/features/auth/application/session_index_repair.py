"""Bounded, idempotent repair of Redis session membership indexes.

The session key is authoritative. An index is only a revocation/discovery
accelerator, so this repair never deletes a session record. Apply mode may
restore a valid orphan membership, while every membership removal is guarded
in Lua by an exact value comparison; a concurrent login that replaces the
record therefore cannot be removed by an older repair observation.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import ValidationError
from redis.exceptions import RedisError, ResponseError

from app.features.auth.adapters.persistence.mongodb.account_repository import (
    _SESSION_KEY,
    _USER_SESSIONS_KEY,
    SessionData,
)
from app.observability.logging import logger

if TYPE_CHECKING:
    from typing import Any

    from redis.asyncio import Redis

_REPAIR_INDEX_SCRIPT = """
local removed = 0
for i = 1, #ARGV - 2, 2 do
    local session_id = ARGV[i]
    local expected = ARGV[i + 1]
    if redis.call('SISMEMBER', KEYS[1], session_id) == 1 then
        local current = redis.call('GET', KEYS[2] .. session_id)
        if not current or current == expected then
            removed = removed + redis.call('SREM', KEYS[1], session_id)
        end
    end
end
local members = redis.call('SCARD', KEYS[1])
if members == 0 then
    redis.call('DEL', KEYS[1])
    return {removed, 0, -2}
end
local desired_ttl = tonumber(ARGV[#ARGV])
if desired_ttl > 0 then
    local current_ttl = redis.call('TTL', KEYS[1])
if current_ttl >= 0 and current_ttl < desired_ttl then
        redis.call('EXPIRE', KEYS[1], desired_ttl)
        current_ttl = desired_ttl
    end
    return {removed, 1, current_ttl}
end
return {removed, 1, redis.call('TTL', KEYS[1])}
"""

_REPAIR_ORPHAN_SCRIPT = """
local current = redis.call('GET', KEYS[1])
if not current or current ~= ARGV[1] then
    return {0, -1}
end
local key_type = redis.call('TYPE', KEYS[2]).ok
if key_type ~= 'none' and key_type ~= 'set' then
    return {0, -3}
end
local index_existed = key_type ~= 'none'
redis.call('SADD', KEYS[2], ARGV[2])
local desired_ttl = tonumber(ARGV[3])
local current_ttl = redis.call('TTL', KEYS[2])
if (not index_existed and current_ttl < desired_ttl) or (index_existed and current_ttl >= 0 and current_ttl < desired_ttl) then
    redis.call('EXPIRE', KEYS[2], desired_ttl)
    current_ttl = desired_ttl
end
return {1, current_ttl}
"""


@dataclass(frozen=True)
class IndexRepair:
    """The observed and proposed state for one user-session index."""

    key: str
    user_id: str
    ttl_before: int
    ttl_after: int
    members_before: int
    live_members: int
    stale_members: int
    malformed_members: int
    removed_members: int = 0
    extended: bool = False
    deferred: bool = False
    error: str | None = None


@dataclass
class RepairSummary:
    """Serializable operational result for one bounded repair invocation."""

    dry_run: bool
    scanned_indexes: int = 0
    repaired_indexes: int = 0
    deferred_indexes: int = 0
    ttl_violations: int = 0
    memberships_removed: int = 0
    orphan_memberships_restored: int = 0
    malformed_records: int = 0
    orphan_sessions: int = 0
    failures: int = 0
    timed_out: int = 0
    complete: bool = False
    next_cursor: int = 0
    next_orphan_cursor: int = 0
    indexes: list[IndexRepair] = field(default_factory=list)
    orphan_session_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly report suitable for operators and CI."""
        return {
            "dry_run": self.dry_run,
            "scanned_indexes": self.scanned_indexes,
            "repaired_indexes": self.repaired_indexes,
            "deferred_indexes": self.deferred_indexes,
            "ttl_violations": self.ttl_violations,
            "memberships_removed": self.memberships_removed,
            "orphan_memberships_restored": self.orphan_memberships_restored,
            "malformed_records": self.malformed_records,
            "orphan_sessions": self.orphan_sessions,
            "failures": self.failures,
            "timed_out": self.timed_out,
            "complete": self.complete,
            "next_cursor": self.next_cursor,
            "next_orphan_cursor": self.next_orphan_cursor,
            "indexes": [item.__dict__ for item in self.indexes],
            "orphan_session_ids": self.orphan_session_ids,
        }


class SessionIndexRepair:
    """Run a bounded repair against one explicitly namespaced Redis client."""

    def __init__(
        self,
        redis: Redis,
        *,
        key_prefix: str = "",
        batch_limit: int = 100,
        member_limit: int = 1_000,
        scan_count: int = 100,
        operation_timeout: float = 2.0,
    ) -> None:
        if batch_limit < 1 or member_limit < 1 or scan_count < 1:
            message = "repair limits must be positive"
            raise ValueError(message)
        if operation_timeout <= 0:
            message = "operation_timeout must be positive"
            raise ValueError(message)
        if any(character in key_prefix for character in "*?[]"):
            message = "key_prefix must not contain Redis glob characters"
            raise ValueError(message)
        self._redis = redis
        self._key_prefix = key_prefix
        self._batch_limit = batch_limit
        self._member_limit = member_limit
        self._scan_count = scan_count
        self._operation_timeout = operation_timeout

    @property
    def index_pattern(self) -> str:
        return f"{self._key_prefix}{_USER_SESSIONS_KEY.format('*')}"

    @property
    def session_pattern(self) -> str:
        return f"{self._key_prefix}{_SESSION_KEY.format('*')}"

    def _session_key(self, session_id: str) -> str:
        return f"{self._key_prefix}{_SESSION_KEY.format(session_id)}"

    def _user_index_key(self, user_id: str) -> str:
        return f"{self._key_prefix}{_USER_SESSIONS_KEY.format(user_id)}"

    @staticmethod
    def _text(value: object) -> str:
        return value.decode() if isinstance(value, bytes) else str(value)

    @staticmethod
    def _remaining_ttl(session: SessionData, now: datetime) -> int:
        expires_at = session.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)
        return max(0, math.ceil((expires_at - now).total_seconds()))

    async def _inspect_index(  # noqa: PLR0914, PLR0915
        self, key: str, *, dry_run: bool
    ) -> IndexRepair:
        async with asyncio.timeout(self._operation_timeout):
            ttl_before = int(await self._redis.ttl(key))
            user_id = key.rsplit(":", 1)[-1]
            member_count = int(await self._redis.scard(key))
            if member_count > self._member_limit:
                return IndexRepair(
                    key=key,
                    user_id=user_id,
                    ttl_before=ttl_before,
                    ttl_after=ttl_before,
                    members_before=member_count,
                    live_members=0,
                    stale_members=0,
                    malformed_members=0,
                    deferred=True,
                )

            raw_ids: list[str] = []
            member_cursor = 0
            while True:
                member_cursor, page = await self._redis.sscan(
                    key,
                    cursor=member_cursor,
                    count=min(self._scan_count, self._member_limit - len(raw_ids) + 1),
                )
                raw_ids.extend(self._text(item) for item in page)
                if len(raw_ids) > self._member_limit:
                    return IndexRepair(
                        key=key,
                        user_id=user_id,
                        ttl_before=ttl_before,
                        ttl_after=ttl_before,
                        members_before=len(raw_ids),
                        live_members=0,
                        stale_members=0,
                        malformed_members=0,
                        deferred=True,
                    )
                if member_cursor == 0:
                    break
            raw_ids.sort()

            async with self._redis.pipeline(transaction=False) as pipe:
                for session_id in raw_ids:
                    pipe.get(self._session_key(str(session_id)))
                raw_values = await pipe.execute()

            now = datetime.now(UTC)
            stale_pairs: list[tuple[str, str]] = []
            live_members = 0
            malformed_members = 0
            max_ttl = 0
            for session_id, raw in zip(raw_ids, raw_values, strict=True):
                sid = self._text(session_id)
                raw_text = "" if raw is None else self._text(raw)
                if raw is None:
                    stale_pairs.append((sid, raw_text))
                    continue
                try:
                    session = SessionData.model_validate_json(raw_text)
                except ValidationError:
                    malformed_members += 1
                    stale_pairs.append((sid, raw_text))
                    continue
                if session.session_id != sid or session.user_id != user_id:
                    malformed_members += 1
                    stale_pairs.append((sid, raw_text))
                    continue
                remaining = self._remaining_ttl(session, now)
                if remaining == 0:
                    stale_pairs.append((sid, raw_text))
                    continue
                live_members += 1
                max_ttl = max(max_ttl, remaining)

            ttl_after = ttl_before if ttl_before == -1 else max(ttl_before, max_ttl)
            repair = IndexRepair(
                key=key,
                user_id=user_id,
                ttl_before=ttl_before,
                ttl_after=ttl_after,
                members_before=len(raw_ids),
                live_members=live_members,
                stale_members=len(stale_pairs),
                malformed_members=malformed_members,
                extended=ttl_before >= 0 and max_ttl > ttl_before,
            )
            if dry_run:
                return repair

            argv: list[str] = []
            for session_id, raw_text in stale_pairs:
                argv.extend((session_id, raw_text))
            argv.append(str(max_ttl))
            result = await self._redis.eval(
                _REPAIR_INDEX_SCRIPT,
                2,
                key,
                f"{self._key_prefix}{_SESSION_KEY.split('{}', 1)[0]}",
                *argv,
            )
            removed = int(result[0])
            final_exists = int(result[1]) == 1
            final_ttl = int(result[2])
            return IndexRepair(
                key=repair.key,
                user_id=repair.user_id,
                ttl_before=repair.ttl_before,
                ttl_after=final_ttl if final_exists else -2,
                members_before=repair.members_before,
                live_members=repair.live_members,
                stale_members=repair.stale_members,
                malformed_members=repair.malformed_members,
                removed_members=removed,
                extended=final_ttl > ttl_before,
                deferred=repair.deferred,
                error=repair.error,
            )

    async def _scan_orphans(  # noqa: PLR0912, PLR0915
        self,
        *,
        cursor: int,
        limit: int,
        dry_run: bool,
        summary: RepairSummary,
    ) -> None:
        current = cursor
        checked = 0
        while checked < limit:
            try:
                current, keys = await self._redis.scan(
                    cursor=current,
                    match=self.session_pattern,
                    count=min(self._scan_count, limit - checked),
                )
            except RedisError as exc:
                summary.failures += 1
                summary.next_orphan_cursor = int(current)
                logger.bind(error=str(exc), cursor=current).warning(
                    "auth_session_index_orphan_scan_failed"
                )
                return
            if not keys:
                if current == 0:
                    break
                continue
            try:
                async with self._redis.pipeline(transaction=False) as pipe:
                    for key in keys:
                        pipe.get(key)
                    raw_values = await pipe.execute(raise_on_error=False)
            except RedisError as exc:
                summary.failures += 1
                summary.next_orphan_cursor = int(current)
                logger.bind(error=str(exc), cursor=current).warning(
                    "auth_session_index_orphan_page_failed"
                )
                return
            for _key, raw in zip(keys, raw_values, strict=True):
                checked += 1
                if raw is None:
                    continue
                if isinstance(raw, ResponseError):
                    summary.failures += 1
                    continue
                try:
                    raw_text = self._text(raw)
                    session = SessionData.model_validate_json(raw_text)
                except ValidationError:
                    summary.malformed_records += 1
                    continue
                remaining = self._remaining_ttl(session, datetime.now(UTC))
                if remaining == 0:
                    continue
                index_key = self._user_index_key(session.user_id)
                try:
                    indexed = await self._redis.sismember(index_key, session.session_id)
                except RedisError as exc:
                    summary.failures += 1
                    logger.bind(key=index_key, error=str(exc)).warning(
                        "auth_session_index_orphan_check_failed"
                    )
                    continue
                if not indexed:
                    summary.orphan_sessions += 1
                    if len(summary.orphan_session_ids) < 100:
                        summary.orphan_session_ids.append(session.session_id)
                    if dry_run:
                        continue
                    try:
                        async with asyncio.timeout(self._operation_timeout):
                            restored, _ttl = await self._redis.eval(
                                _REPAIR_ORPHAN_SCRIPT,
                                2,
                                self._session_key(session.session_id),
                                index_key,
                                raw_text,
                                session.session_id,
                                remaining,
                            )
                        if int(restored) == 1:
                            summary.orphan_memberships_restored += 1
                    except TimeoutError:
                        summary.timed_out += 1
                        summary.failures += 1
                    except RedisError as exc:
                        summary.failures += 1
                        logger.bind(key=index_key, error=str(exc)).warning(
                            "auth_session_index_orphan_repair_failed"
                        )
            if current == 0 or checked >= limit:
                break
        summary.next_orphan_cursor = int(current)

    async def run(
        self,
        *,
        cursor: int = 0,
        dry_run: bool = True,
        orphan_cursor: int = 0,
        orphan_limit: int = 0,
    ) -> RepairSummary:
        """Repair at most ``batch_limit`` indexes and return a resumable report."""
        if cursor < 0 or orphan_cursor < 0 or orphan_limit < 0:
            message = "cursors and limits must not be negative"
            raise ValueError(message)
        summary = RepairSummary(dry_run=dry_run)
        current = cursor
        try:
            while summary.scanned_indexes < self._batch_limit:
                current, keys = await self._redis.scan(
                    cursor=current,
                    match=self.index_pattern,
                    count=min(self._scan_count, self._batch_limit - summary.scanned_indexes),
                )
                # Redis treats COUNT as a hint and can return more than the
                # requested page size. Process the complete page rather than
                # advancing the cursor past unprocessed keys.
                for key in keys:
                    summary.scanned_indexes += 1
                    try:
                        repair = await self._inspect_index(str(key), dry_run=dry_run)
                    except TimeoutError:
                        summary.timed_out += 1
                        summary.failures += 1
                        repair = IndexRepair(
                            key=str(key),
                            user_id=str(key).rsplit(":", 1)[-1],
                            ttl_before=-3,
                            ttl_after=-3,
                            members_before=0,
                            live_members=0,
                            stale_members=0,
                            malformed_members=0,
                            error="operation timed out",
                        )
                    except (RedisError, ValueError, ValidationError) as exc:
                        summary.failures += 1
                        repair = IndexRepair(
                            key=str(key),
                            user_id=str(key).rsplit(":", 1)[-1],
                            ttl_before=-3,
                            ttl_after=-3,
                            members_before=0,
                            live_members=0,
                            stale_members=0,
                            malformed_members=0,
                            error=str(exc),
                        )
                    summary.indexes.append(repair)
                    summary.malformed_records += repair.malformed_members
                    summary.memberships_removed += repair.removed_members
                    if repair.ttl_before >= 0 and repair.ttl_after > repair.ttl_before:
                        summary.ttl_violations += 1
                    if repair.deferred:
                        summary.deferred_indexes += 1
                    elif repair.error is None:
                        summary.repaired_indexes += 1
                if current == 0 or summary.scanned_indexes >= self._batch_limit:
                    break
            summary.next_cursor = int(current)
            summary.complete = current == 0
            if orphan_limit:
                await self._scan_orphans(
                    cursor=orphan_cursor,
                    limit=orphan_limit,
                    dry_run=dry_run,
                    summary=summary,
                )
        except (TimeoutError, RedisError) as exc:
            summary.failures += 1
            logger.bind(error=str(exc), dry_run=dry_run).warning("auth_session_index_repair_failed")
        logger.bind(**summary.as_dict()).info("auth_session_index_repair_completed")
        return summary
