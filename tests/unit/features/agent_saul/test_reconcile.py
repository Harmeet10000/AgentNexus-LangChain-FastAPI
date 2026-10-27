"""Thread-lock fencing contracts for the transcript reconciliation sweep."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

import pytest
from returns.result import Success

from app.features.agent_saul import reconcile as reconcile_module
from app.features.agent_saul.reconcile import (
    _reconcile_hinted_threads,
    _reconcile_locked,
    _reconcile_stale_sessions,
    _ThreadReconcileOutcome,
)
from app.features.agent_saul.transcript.projection import Projection, SessionStatus, SnapshotView
from app.features.agent_saul.transcript.repository import (
    ApplyOutcome,
    SessionMeta,
    TranscriptTurnContext,
)
from app.features.agent_saul.transcript.sync import PreparedSnapshot
from app.observability.logging import logger

if TYPE_CHECKING:
    from typing import Any

pytestmark = pytest.mark.unit

NOW = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)


class _LockDouble:
    def __init__(self, acquired: bool) -> None:
        self.acquired = acquired
        self.released = False

    async def acquire(self) -> bool:
        return self.acquired

    async def release(self) -> None:
        self.released = True


class _RedisDouble:
    def __init__(self, hint_ids: list[str] | None = None) -> None:
        self.hint_ids = hint_ids

    async def spop(self, _key: str, _count: int) -> list[str] | None:
        return self.hint_ids


class _RepoDouble:
    def __init__(self, sessions: tuple[SessionMeta, ...] = ()) -> None:
        self.sessions = sessions

    async def list_stale_sessions(self, **_kwargs: Any) -> Success[tuple[SessionMeta, ...]]:
        return Success(self.sessions)


class _ProjectionRepoDouble:
    def __init__(self, owner_user_id: str) -> None:
        self.owner_user_id = owner_user_id
        self.apply_calls = 0

    async def get_sweep_context(
        self,
        *,
        session_id: str,
    ) -> Success[tuple[str, TranscriptTurnContext]]:
        assert session_id == "thread"
        return Success((self.owner_user_id, TranscriptTurnContext(last_checkpoint_id="prior")))

    async def apply_projection(self, *_args: Any, **_kwargs: Any) -> Success[ApplyOutcome]:
        self.apply_calls += 1
        return Success(ApplyOutcome(applied=True))


class _GraphDouble:
    async def aget_state(self, _config: Any) -> object:
        return object()


async def test_reconcile_locked_skips_without_reading_when_thread_is_held(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = _LockDouble(acquired=False)
    monkeypatch.setattr(reconcile_module, "ThreadLock", lambda *_args: lock)

    result = await _reconcile_locked(
        repo=cast("Any", object()),
        graph=cast("Any", object()),
        redis=cast("Any", object()),
        session_id="held-thread",
        now=NOW,
    )

    assert result is None
    assert lock.released is False


@pytest.mark.parametrize("raises", [False, True])
async def test_reconcile_locked_releases_after_reconcile_success_or_failure(
    monkeypatch: pytest.MonkeyPatch,
    raises: bool,
) -> None:
    lock = _LockDouble(acquired=True)
    monkeypatch.setattr(reconcile_module, "ThreadLock", lambda *_args: lock)

    async def reconcile_once(**_kwargs: Any) -> _ThreadReconcileOutcome:
        if raises:
            message = "projection failed"
            raise RuntimeError(message)
        return _ThreadReconcileOutcome(status="active", failed=False)

    monkeypatch.setattr(reconcile_module, "_reconcile_once", reconcile_once)
    call = {
        "repo": cast("Any", object()),
        "graph": cast("Any", object()),
        "redis": cast("Any", object()),
        "session_id": "thread",
        "now": NOW,
    }

    if raises:
        with pytest.raises(RuntimeError, match="projection failed"):
            await _reconcile_locked(**call)
    else:
        result = await _reconcile_locked(**call)
        assert result == _ThreadReconcileOutcome(status="active", failed=False)
    assert lock.released is True


async def test_stale_sweep_does_not_close_a_live_locked_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = _LockDouble(acquired=False)
    monkeypatch.setattr(reconcile_module, "ThreadLock", lambda *_args: lock)
    repo = _RepoDouble((SessionMeta(session_id="live-thread", user_id="owner", status="active"),))

    scanned, closed = await _reconcile_stale_sessions(
        repo=cast("Any", repo),
        graph=cast("Any", object()),
        redis=cast("Any", object()),
        now=NOW,
        batch_size=1,
    )

    assert (scanned, closed) == (1, 0)


async def test_hint_and_stale_logs_preserve_exact_noop_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reason = "noop:heartbeat_stale_within_grace"
    outcome = _ThreadReconcileOutcome(reason=reason, clean=True, failed=False)

    async def reconcile_locked(**_kwargs: Any) -> _ThreadReconcileOutcome:
        return outcome

    monkeypatch.setattr(reconcile_module, "_reconcile_locked", reconcile_locked)
    sink: list[str] = []
    sink_id = logger.add(sink.append, format="{message} | {extra}", level="DEBUG")
    try:
        await _reconcile_hinted_threads(
            repo=cast("Any", object()),
            graph=cast("Any", object()),
            redis=cast("Any", _RedisDouble(["hint-thread"])),
            now=NOW,
            bound=1,
        )
        await _reconcile_stale_sessions(
            repo=cast(
                "Any",
                _RepoDouble(
                    (SessionMeta(session_id="stale-thread", user_id="owner", status="active"),)
                ),
            ),
            graph=cast("Any", object()),
            redis=cast("Any", object()),
            now=NOW,
            batch_size=1,
        )
    finally:
        logger.remove(sink_id)

    assert any("transcript_sweep_hint_reconciled" in line and reason in line for line in sink)
    assert any("transcript_sweep_stale_reconciled" in line and reason in line for line in sink)


async def test_locked_reconcile_rejects_snapshot_owner_mismatch_without_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = _LockDouble(acquired=True)
    repo = _ProjectionRepoDouble(owner_user_id="repository-owner")
    monkeypatch.setattr(reconcile_module, "ThreadLock", lambda *_args: lock)
    monkeypatch.setattr(
        reconcile_module,
        "snapshot_to_view",
        lambda _snapshot, **_kwargs: SnapshotView(user_id="checkpoint-owner", checkpoint_id="cp"),
    )

    def should_not_prepare(**_kwargs: Any) -> PreparedSnapshot:
        pytest.fail("owner mismatch must be rejected before projection")

    monkeypatch.setattr(reconcile_module, "prepare_snapshot", should_not_prepare)

    outcome = await _reconcile_locked(
        repo=cast("Any", repo),
        graph=cast("Any", _GraphDouble()),
        redis=cast("Any", object()),
        session_id="thread",
        now=NOW,
    )

    assert outcome == _ThreadReconcileOutcome(reason="snapshot_owner_mismatch")
    assert repo.apply_calls == 0
    assert lock.released is True


@pytest.mark.parametrize("snapshot_user_id", ["repository-owner", ""])
async def test_locked_reconcile_accepts_matching_or_legacy_empty_snapshot_owner(
    monkeypatch: pytest.MonkeyPatch,
    snapshot_user_id: str,
) -> None:
    lock = _LockDouble(acquired=True)
    repo = _ProjectionRepoDouble(owner_user_id="repository-owner")
    monkeypatch.setattr(reconcile_module, "ThreadLock", lambda *_args: lock)
    monkeypatch.setattr(
        reconcile_module,
        "snapshot_to_view",
        lambda _snapshot, **_kwargs: SnapshotView(user_id=snapshot_user_id, checkpoint_id="cp"),
    )
    monkeypatch.setattr(
        reconcile_module,
        "prepare_snapshot",
        lambda **_kwargs: PreparedSnapshot(
            projection=Projection(
                session_status=SessionStatus.ACTIVE,
                watermark_checkpoint_id="cp",
            )
        ),
    )

    outcome = await _reconcile_locked(
        repo=cast("Any", repo),
        graph=cast("Any", _GraphDouble()),
        redis=cast("Any", object()),
        session_id="thread",
        now=NOW,
    )

    assert outcome == _ThreadReconcileOutcome(
        status=SessionStatus.ACTIVE.value,
        checkpoint_id="cp",
        failed=False,
    )
    assert repo.apply_calls == 1
    assert lock.released is True
