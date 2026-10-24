"""Durable reconciliation of role-change session revocation markers.

The request path remains the security boundary: a pending marker and the
authorization epoch prevent stale sessions from authorizing work.  This
module is only eventual cleanup.  It is intentionally scheduler-agnostic so
Celery, the recovery command, and tests share the same fencing and retry
semantics.
"""

from __future__ import annotations

import json
import secrets
import time
from asyncio import timeout
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol, cast

from beanie import PydanticObjectId
from returns.result import Failure

from app.connections.mongodb import close_mongo_client, create_mongo_client
from app.features.auth.models import User
from app.features.auth.repository import RefreshTokenRepository
from app.features.auth.token_audit_log import TokenAuditLog
from app.features.users.repository import UserAdminRepository
from app.observability.logging import logger

if TYPE_CHECKING:
    from typing import Any

    from motor.motor_asyncio import AsyncIOMotorClient
    from redis.asyncio import Redis
    from returns.result import Result

    from app.config import Settings
    from app.features.auth.exceptions import AuthError
    from app.features.users.exceptions import UsersError


class _UserLike(Protocol):
    id: object
    role_revision: int
    session_revocation_pending: bool
    updated_at: datetime
    session_revocation_pending_at: datetime | None


type UserResult = Result[_UserLike, AuthError | UsersError]
type ListUsers = Callable[[int, int], Awaitable[list[_UserLike]]]
type GetUser = Callable[[str], Awaitable[_UserLike | None]]
type RevokeSessions = Callable[[str, int], Awaitable[UserResult]]
type ClearMarker = Callable[[_UserLike, int], Awaitable[UserResult]]
type CountUsers = Callable[[], Awaitable[int]]

_LEASE_PREFIX = "auth:session_revocation:lease:"
_RETRY_PREFIX = "auth:session_revocation:retry:"
_RELEASE_LEASE_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
end
return 0
"""
_SET_RETRY_IF_OWNER_SCRIPT = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then
    return 0
end
return redis.call('SET', KEYS[2], ARGV[2])
"""
_DELETE_RETRY_IF_OWNER_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[2])
end
return 0
"""


@dataclass(frozen=True, slots=True)
class PendingRevocationConfig:
    """Bounded operational policy for one reconciliation pass."""

    batch_size: int = 50
    lease_seconds: int = 120
    max_attempts: int = 8
    backoff_base_seconds: int = 15
    backoff_max_seconds: int = 900
    jitter_seconds: int = 5
    oldest_alert_seconds: int = 300
    max_runtime_seconds: int = 45


@dataclass(frozen=True, slots=True)
class RetryState:
    revision: int
    attempts: int = 0
    next_attempt_at: float = 0
    last_error_category: str | None = None
    poisoned: bool = False

    @classmethod
    def from_json(cls, raw: str | bytes | None, revision: int) -> RetryState:
        if raw is None:
            return cls(revision=revision)
        try:
            value = json.loads(raw)
            if not isinstance(value, dict) or int(value.get("revision", -1)) != revision:
                return cls(revision=revision)
            return cls(
                revision=revision,
                attempts=max(0, int(value.get("attempts", 0))),
                next_attempt_at=max(0.0, float(value.get("next_attempt_at", 0))),
                last_error_category=(
                    str(value["last_error_category"])
                    if value.get("last_error_category") is not None
                    else None
                ),
                poisoned=bool(value.get("poisoned", False)),
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            # A malformed operational record must never block security cleanup.
            return cls(revision=revision)

    def encode(self) -> str:
        return json.dumps(
            {
                "revision": self.revision,
                "attempts": self.attempts,
                "next_attempt_at": self.next_attempt_at,
                "last_error_category": self.last_error_category,
                "poisoned": self.poisoned,
            },
            separators=(",", ":"),
        )


@dataclass(frozen=True, slots=True)
class ReconciliationReport:
    scanned: int = 0
    attempted: int = 0
    succeeded: int = 0
    skipped: int = 0
    stale_revision: int = 0
    retryable_failure: int = 0
    permanent_failure: int = 0
    poisoned: int = 0
    oldest_pending_age_seconds: float = 0
    alert_oldest_pending: bool = False
    queue_depth: int = 0
    preview: tuple[dict[str, object], ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "scanned": self.scanned,
            "attempted": self.attempted,
            "succeeded": self.succeeded,
            "skipped": self.skipped,
            "stale_revision": self.stale_revision,
            "retryable_failure": self.retryable_failure,
            "permanent_failure": self.permanent_failure,
            "poisoned": self.poisoned,
            "oldest_pending_age_seconds": self.oldest_pending_age_seconds,
            "alert_oldest_pending": self.alert_oldest_pending,
            "queue_depth": self.queue_depth,
            "preview": list(self.preview),
        }


class PendingRevocationReconciler:
    """Process a bounded snapshot with Redis leases and revision fencing."""

    def __init__(
        self,
        redis: Redis,
        *,
        list_pending: ListUsers,
        get_user: GetUser,
        revoke_sessions: RevokeSessions,
        clear_marker: ClearMarker,
        count_pending: CountUsers | None = None,
        config: PendingRevocationConfig | None = None,
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
        random: Callable[[int], int] = secrets.randbelow,
    ) -> None:
        self._redis = redis
        self._list_pending = list_pending
        self._get_user = get_user
        self._revoke_sessions = revoke_sessions
        self._clear_marker = clear_marker
        self._count_pending = count_pending
        self._config = config or PendingRevocationConfig()
        self._clock = clock
        self._monotonic = monotonic
        self._random = random

    async def run(  # noqa: PLR0912 — bounded orchestration handles explicit failure states
        self,
        *,
        user_id: str | None = None,
        dry_run: bool = False,
        force: bool = False,
    ) -> dict[str, object]:
        """Reconcile at most one bounded batch; safe to invoke on every beat."""
        started = self._monotonic()
        deadline = started + self._config.max_runtime_seconds
        offset = 0
        try:
            offset_raw = await self._await_budget(
                self._redis.get("auth:session_revocation:cursor"), deadline
            )
            offset_value = cast("str | bytes | None", offset_raw)
            offset = max(0, int(offset_value or 0))
        except Exception as exc:  # noqa: BLE001 — Redis outage must not clear markers
            exc.add_note("operation=pending_reconcile, stage=cursor_read")
        if user_id is not None:
            selected = await self._get_user_with_deadline(user_id, deadline)
            users = [selected] if selected is not None else []
        else:
            users = await self._list_pending_with_deadline(self._config.batch_size, offset, deadline)
        queue_depth = len(users)
        if self._count_pending is not None:
            try:
                queue_depth = await self._count_with_deadline(deadline)
            except Exception as exc:  # noqa: BLE001 — depth is advisory; cleanup can continue
                exc.add_note("operation=pending_reconcile, stage=queue_depth")
        # The work page can start after the oldest entries because the cursor
        # deliberately advances past deferred/poisoned users. Read the head
        # separately so the alert describes the queue, not just this page.
        oldest_users = users if offset == 0 else []
        if not oldest_users and user_id is None and self._monotonic() < deadline:
            try:
                oldest_users = await self._list_pending_with_deadline(1, 0, deadline)
            except Exception as exc:  # noqa: BLE001 — alerting is advisory
                exc.add_note("operation=pending_reconcile, stage=oldest_pending_read")
        oldest = self._oldest_age(oldest_users)
        report = ReconciliationReport(
            scanned=len(users),
            oldest_pending_age_seconds=oldest,
            alert_oldest_pending=oldest >= self._config.oldest_alert_seconds,
            queue_depth=queue_depth,
        )
        if report.alert_oldest_pending:
            logger.bind(
                oldest_pending_age_seconds=oldest,
                queue_depth=queue_depth,
            ).warning("auth_pending_revocation_oldest_alert")

        for user in users:
            if self._monotonic() >= deadline:
                break
            if not user.session_revocation_pending:
                continue
            if dry_run:
                state = await self._read_retry_state(str(user.id), user.role_revision, deadline)
                preview = (*report.preview, {
                    "user_id": str(user.id),
                    "revision": user.role_revision,
                    "action": (
                        "deferred"
                        if (state.poisoned or state.next_attempt_at > self._clock()) and not force
                        else "would_revoke"
                    ),
                })
                report = replace(report, preview=preview)
            result = await self._process_user(
                user, dry_run=dry_run, force=force, deadline=deadline
            )
            report = replace(
                report,
                attempted=report.attempted + result.get("attempted", 0),
                succeeded=report.succeeded + result.get("succeeded", 0),
                skipped=report.skipped + result.get("skipped", 0),
                stale_revision=report.stale_revision + result.get("stale_revision", 0),
                retryable_failure=report.retryable_failure + result.get("retryable_failure", 0),
                permanent_failure=report.permanent_failure + result.get("permanent_failure", 0),
                poisoned=report.poisoned + result.get("poisoned", 0),
            )
        if user_id is None and not dry_run:
            next_offset = offset + len(users) if len(users) >= self._config.batch_size else 0
            try:
                await self._await_budget(
                    self._redis.set(
                        "auth:session_revocation:cursor",
                        str(next_offset),
                        ex=self._config.backoff_max_seconds * 4,
                    ),
                    deadline,
                )
            except Exception as exc:  # noqa: BLE001 — cursor is advisory only
                exc.add_note("operation=pending_reconcile, stage=cursor_write")
        logger.bind(**report.as_dict()).info("auth_pending_revocation_reconciliation_finished")
        return report.as_dict()

    def _oldest_age(self, users: list[_UserLike]) -> float:
        # A legacy marker has no timestamp. Treat it as oldest rather than
        # silently allowing it to disappear from the operational alert.
        pending = [
            getattr(u, "session_revocation_pending_at", None)
            or datetime.fromtimestamp(0, UTC)
            for u in users
            if u.session_revocation_pending
        ]
        if not pending:
            return 0
        oldest = min(pending)
        if oldest.tzinfo is None:
            oldest = oldest.replace(tzinfo=UTC)
        return max(0.0, self._clock() - oldest.timestamp())

    async def _process_user(  # noqa: PLR0912 — lease/retry state machine branches are explicit
        self,
        user: _UserLike,
        *,
        dry_run: bool,
        force: bool,
        deadline: float,
    ) -> dict[str, int]:
        user_id = str(user.id)
        revision = user.role_revision
        retry_key = _RETRY_PREFIX + user_id
        lease_key = _LEASE_PREFIX + user_id
        try:
            state = await self._read_retry_state(user_id, revision, deadline)
            if not force and (state.poisoned or state.next_attempt_at > self._clock()):
                return {"skipped": 1}
            if dry_run:
                return {"attempted": 0, "skipped": 1}
            lease_token = secrets.token_urlsafe(18)
            claimed = await self._await_budget(
                self._redis.set(
                    lease_key,
                    lease_token,
                    nx=True,
                    ex=self._config.lease_seconds,
                ),
                deadline,
            )
            if not claimed:
                return {"skipped": 1}
            try:
                # Re-read state after the lease: another worker may have just
                # scheduled a backoff or marked this revision poisoned.
                state = await self._read_retry_state(user_id, revision, deadline)
                if not force and (state.poisoned or state.next_attempt_at > self._clock()):
                    return {"skipped": 1}
                try:
                    current = await self._get_user_with_deadline(user_id, deadline)
                except Exception as exc:  # noqa: BLE001 — retry policy handles Mongo outage
                    return await self._record_exception(
                        retry_key,
                        state,
                        exc,
                        lease_key=lease_key,
                        lease_token=lease_token,
                        deadline=deadline,
                    )
                if (
                    current is None
                    or not current.session_revocation_pending
                    or current.role_revision != revision
                ):
                    return {"stale_revision": 1}
                lease_owner = await self._await_budget(self._redis.get(lease_key), deadline)
                if lease_owner != lease_token:
                    return {"skipped": 1}
                try:
                    revoke = await self._revoke_sessions_with_deadline(user_id, revision, deadline)
                except Exception as exc:  # noqa: BLE001 — adapter boundary
                    return await self._record_exception(
                        retry_key,
                        state,
                        exc,
                        lease_key=lease_key,
                        lease_token=lease_token,
                        deadline=deadline,
                    )
                if isinstance(revoke, Failure):
                    return await self._record_failure(
                        retry_key,
                        state,
                        revoke.failure(),
                        lease_key=lease_key,
                        lease_token=lease_token,
                        deadline=deadline,
                    )
                try:
                    cleared = await self._clear_marker_with_deadline(current, revision, deadline)
                except Exception as exc:  # noqa: BLE001 — adapter boundary
                    return await self._record_exception(
                        retry_key,
                        state,
                        exc,
                        lease_key=lease_key,
                        lease_token=lease_token,
                        deadline=deadline,
                    )
                if isinstance(cleared, Failure):
                    return await self._record_failure(
                        retry_key,
                        state,
                        cleared.failure(),
                        lease_key=lease_key,
                        lease_token=lease_token,
                        deadline=deadline,
                    )
                after = cleared.unwrap()
                if after.session_revocation_pending or after.role_revision != revision:
                    logger.bind(user_id=user_id, revision=revision).info(
                        "auth_pending_revocation_stale_revision"
                    )
                    return {"stale_revision": 1}
                await self._await_budget(
                    self._redis.eval(
                        _DELETE_RETRY_IF_OWNER_SCRIPT,
                        2,
                        lease_key,
                        retry_key,
                        lease_token,
                    ),
                    deadline,
                )
                logger.bind(user_id=user_id, revision=revision).info(
                    "auth_pending_revocation_succeeded"
                )
                return {"attempted": 1, "succeeded": 1}
            finally:
                try:
                    await self._await_budget(
                        self._redis.eval(_RELEASE_LEASE_SCRIPT, 1, lease_key, lease_token),
                        deadline,
                    )
                except Exception as exc:  # noqa: BLE001 — lease TTL is the final fence
                    exc.add_note("operation=pending_reconcile, stage=lease_release")
                    logger.bind(user_id=user_id).warning(
                        "auth_pending_revocation_lease_release_deferred"
                    )
        except Exception as exc:  # noqa: BLE001 — Redis/Mongo boundary; marker remains set
            exc.add_note(f"user_id={user_id}, revision={revision}, operation=pending_reconcile")
            logger.bind(
                user_id=user_id, revision=revision, error_category=type(exc).__name__
            ).warning("auth_pending_revocation_retryable_failure")
            return {"attempted": 1, "retryable_failure": 1}

    async def _read_retry_state(self, user_id: str, revision: int, deadline: float) -> RetryState:
        raw = await self._await_budget(self._redis.get(_RETRY_PREFIX + user_id), deadline)
        return RetryState.from_json(cast("str | bytes | None", raw), revision)

    async def _get_user_with_deadline(self, user_id: str, deadline: float) -> _UserLike | None:
        value = await self._await_budget(self._get_user(user_id), deadline)
        return cast("_UserLike | None", value)

    async def _list_pending_with_deadline(
        self, limit: int, offset: int, deadline: float
    ) -> list[_UserLike]:
        value = await self._await_budget(self._list_pending(limit, offset), deadline)
        return cast("list[_UserLike]", value)

    async def _count_with_deadline(self, deadline: float) -> int:
        count_pending = self._count_pending
        if count_pending is None:
            return 0
        value = await self._await_budget(count_pending(), deadline)
        return cast("int", value)

    async def _revoke_sessions_with_deadline(
        self, user_id: str, revision: int, deadline: float
    ) -> UserResult:
        value = await self._await_budget(self._revoke_sessions(user_id, revision), deadline)
        return cast("UserResult", value)

    async def _clear_marker_with_deadline(
        self, user: _UserLike, revision: int, deadline: float
    ) -> UserResult:
        value = await self._await_budget(self._clear_marker(user, revision), deadline)
        return cast("UserResult", value)

    async def _await_budget(self, awaitable: Awaitable[object], deadline: float) -> object:
        remaining = deadline - self._monotonic()
        if remaining <= 0:
            message = "pending revocation pass exceeded its runtime budget"
            raise TimeoutError(message)
        async with timeout(remaining):
            return await awaitable

    async def _record_failure(
        self,
        retry_key: str,
        state: RetryState,
        error: AuthError | UsersError,
        *,
        lease_key: str,
        lease_token: str,
        deadline: float,
    ) -> dict[str, int]:
        return await self._record_failure_details(
            retry_key,
            state,
            type(error).__name__,
            bool(getattr(error, "retryable", True)),
            lease_key=lease_key,
            lease_token=lease_token,
            deadline=deadline,
        )

    async def _record_exception(
        self,
        retry_key: str,
        state: RetryState,
        error: Exception,
        *,
        lease_key: str,
        lease_token: str,
        deadline: float,
    ) -> dict[str, int]:
        try:
            return await self._record_failure_details(
                retry_key,
                state,
                type(error).__name__,
                True,
                lease_key=lease_key,
                lease_token=lease_token,
                deadline=deadline,
            )
        except Exception as record_error:  # noqa: BLE001 — Redis may be the failed dependency
            record_error.add_note("operation=pending_reconcile, stage=retry_state_write")
            logger.bind(error_category=type(error).__name__).warning(
                "auth_pending_revocation_retry_state_unavailable"
            )
            return {"attempted": 1, "retryable_failure": 1}

    async def _record_failure_details(
        self,
        retry_key: str,
        state: RetryState,
        category: str,
        retryable: bool,
        *,
        lease_key: str,
        lease_token: str,
        deadline: float,
    ) -> dict[str, int]:
        attempts = state.attempts + 1
        poisoned = (not retryable) or attempts >= self._config.max_attempts
        if poisoned:
            next_attempt = 0
        else:
            delay = min(
                self._config.backoff_max_seconds,
                self._config.backoff_base_seconds * (2 ** (attempts - 1)),
            )
            jitter = (
                self._random(self._config.jitter_seconds + 1) if self._config.jitter_seconds else 0
            )
            next_attempt = self._clock() + delay + jitter
        next_state = RetryState(
            revision=state.revision,
            attempts=attempts,
            next_attempt_at=next_attempt,
            last_error_category=category,
            poisoned=poisoned,
        )
        # Retry state is durable until success. More importantly, the lease
        # token fences a slow worker whose lease expired and was reacquired by
        # another worker; it must never overwrite the newer worker's retry
        # budget or poison state.
        saved = await self._await_budget(
            self._redis.eval(
                _SET_RETRY_IF_OWNER_SCRIPT,
                2,
                lease_key,
                retry_key,
                lease_token,
                next_state.encode(),
            ),
            deadline,
        )
        if not saved:
            logger.bind(user_id=retry_key.removeprefix(_RETRY_PREFIX)).info(
                "auth_pending_revocation_lease_lost_before_retry_write"
            )
            return {"attempted": 1, "retryable_failure": 1}
        logger.bind(
            error_category=category,
            attempts=attempts,
            poisoned=poisoned,
        ).warning(
            "auth_pending_revocation_permanent_failure"
            if poisoned
            else "auth_pending_revocation_retry_scheduled"
        )
        return {
            "attempted": 1,
            "permanent_failure" if not retryable else "retryable_failure": 1,
            "poisoned": int(poisoned),
        }


async def list_pending_users(limit: int, offset: int = 0) -> list[_UserLike]:
    """Bounded, oldest-first Mongo discovery used by workers and the CLI."""
    users = (
        await User.find(User.session_revocation_pending == True)  # noqa: E712 — Mongo predicate
        .sort("session_revocation_pending_at", "updated_at")
        .skip(offset)
        .limit(limit)
        .to_list()
    )
    return cast("list[_UserLike]", users)


async def count_pending_users() -> int:
    """Return queue depth separately from the bounded work page."""
    return await User.find(User.session_revocation_pending == True).count()  # noqa: E712


async def get_user(user_id: str) -> User | None:
    if not PydanticObjectId.is_valid(user_id):
        return None
    return await User.get(PydanticObjectId(user_id))


async def build_production_reconciler(
    redis: Redis, settings: Settings
) -> PendingRevocationReconciler:
    """Wire the reconciler to Beanie and the existing epoch-fenced repository."""
    # Celery processes do not run the HTTP lifespan.  Initialising Beanie once
    # in the child process makes restart recovery explicit and keeps every task
    # invocation on the same Mongo pool.
    global _mongo_client
    if _mongo_client is None:
        _mongo_client, _ = await create_mongo_client(
            uri=settings.MONGODB_URI,
            db_name=settings.MONGODB_DB_NAME,
            document_models=[User, TokenAuditLog],
        )
    config = PendingRevocationConfig(
        batch_size=settings.AUTH_PENDING_REVOCATION_BATCH_SIZE,
        lease_seconds=settings.AUTH_PENDING_REVOCATION_LEASE_SECONDS,
        max_attempts=settings.AUTH_PENDING_REVOCATION_MAX_ATTEMPTS,
        backoff_base_seconds=settings.AUTH_PENDING_REVOCATION_BACKOFF_BASE_SECONDS,
        backoff_max_seconds=settings.AUTH_PENDING_REVOCATION_BACKOFF_MAX_SECONDS,
        jitter_seconds=settings.AUTH_PENDING_REVOCATION_BACKOFF_JITTER_SECONDS,
        oldest_alert_seconds=settings.AUTH_PENDING_REVOCATION_OLDEST_ALERT_SECONDS,
        max_runtime_seconds=settings.AUTH_PENDING_REVOCATION_MAX_RUNTIME_SECONDS,
    )
    token_repository = RefreshTokenRepository(redis)

    async def revoke(user_id: str, revision: int) -> UserResult:
        return cast(
            "UserResult",
            await token_repository.revoke_all_user_sessions(
                user_id=user_id,
                reason="role_changed_reconciled",
                authz_revision=revision,
            ),
        )

    async def clear(user: _UserLike, revision: int) -> UserResult:
        return cast(
            "UserResult",
            await UserAdminRepository.clear_session_revocation_pending(
                cast("User", user), expected_revision=revision
            ),
        )

    return PendingRevocationReconciler(
        redis,
        list_pending=list_pending_users,
        get_user=get_user,
        revoke_sessions=revoke,
        clear_marker=clear,
        count_pending=count_pending_users,
        config=config,
    )


_mongo_client: AsyncIOMotorClient[Any] | None = None


def close_production_resources() -> None:
    """Close the child-owned Mongo client during Celery worker shutdown."""
    global _mongo_client  # noqa: PLW0603 — process-local worker resource
    if _mongo_client is not None:
        close_mongo_client(_mongo_client)
        _mongo_client = None
