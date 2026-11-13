"""Checkpoint GC teardown outcomes are distinguishable (harden 3.13).

The sweep's retention contract: paused threads are retained 24 h and closed
abandoned_expired; terminal threads are garbage-collected after 7 days via the
public ``adelete_thread`` API, with a pending-audit stamp that makes a crash
between stamp and delete a retryable, idempotent state — never a silent loss.
Each refusal outcome is DISTINCT: generation moved, a newer turn arrived, a
failed catch-up, and a live lock all keep the checkpoint for different reasons.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest
from returns.result import Failure, Success

import app.features.agent_saul.application.reconcile as reconcile_module
from app.features.agent_saul.application.reconcile import (
    TRANSCRIPT_RETENTION,
    _gc_checkpointed_threads,
    _gc_one_session,
    _ThreadReconcileOutcome,
)
from app.features.agent_saul.application.transcript.repository import SessionMeta

if TYPE_CHECKING:
    from typing import Any

SESSION_ID = "session-1"
NOW = datetime(2026, 11, 13, 12, 0, tzinfo=UTC)
OLDER_THAN = NOW - TRANSCRIPT_RETENTION


class _Checkpointer:
    """Records teardown calls against the public adelete_thread API."""

    def __init__(self) -> None:
        self.deleted: list[str] = []

    async def adelete_thread(self, session_id: str) -> None:
        self.deleted.append(session_id)


class _Repo:
    def __init__(
        self,
        *,
        current_id: str | None,
        current_ts: datetime | None,
        stamp_ok: bool = True,
        confirm_ok: bool = True,
        candidate: SessionMeta | None = None,
    ) -> None:
        self._current_id = current_id
        self._current_ts = current_ts
        self._stamp_ok = stamp_ok
        self._confirm_ok = confirm_ok
        self._candidate = candidate
        self.stamps: list[dict[str, Any]] = []
        self.confirms: list[str] = []

    async def peek_session_checkpoint(
        self, *, session_id: str
    ) -> Success[tuple[str | None, datetime | None]]:
        del session_id
        return Success((self._current_id, self._current_ts))

    async def list_gc_candidates(
        self, *, older_than: datetime, limit: int
    ) -> Success[list[SessionMeta]]:
        del older_than, limit
        return Success([self._candidate] if self._candidate is not None else [])

    async def stamp_gc_audit(
        self,
        *,
        session_id: str,
        deleted_checkpoint_id: str | None,
        swept_at: datetime,
        pending: bool,
    ) -> Success[None]:
        del session_id
        self.stamps.append(
            {
                "deleted_checkpoint_id": deleted_checkpoint_id,
                "swept_at": swept_at,
                "pending": pending,
            }
        )
        if self._stamp_ok:
            return Success(None)
        return Failure(ValueError("stamp unavailable"))

    async def confirm_gc_audit(self, *, session_id: str, confirmed_at: datetime) -> Success[None]:
        del confirmed_at
        self.confirms.append(session_id)
        if self._confirm_ok:
            return Success(None)
        return Failure(ValueError("confirm unavailable"))


class _Redis:
    async def ping(self) -> bool:
        return True


def _patch_reconcile(monkeypatch: pytest.MonkeyPatch, outcome: _ThreadReconcileOutcome) -> None:
    async def scripted(**_kwargs: Any) -> _ThreadReconcileOutcome:
        return outcome

    monkeypatch.setattr(reconcile_module, "_reconcile_once", scripted)


def _meta(*, gc_pending: bool = False) -> SessionMeta:
    return SessionMeta(
        session_id=SESSION_ID,
        user_id="user-1",
        status="completed",
        last_checkpoint_id="cp-1",
        last_checkpoint_ts=OLDER_THAN - timedelta(days=1),
        gc_pending=gc_pending,
    )


def _make_lock(monkeypatch: pytest.MonkeyPatch, *, acquired: bool) -> list[str]:
    released: list[str] = []

    class _FakeLock:
        def __init__(self, _redis: object, thread_id: str) -> None:
            self.thread_id = thread_id

        async def acquire(self) -> bool:
            return acquired

        async def release(self) -> None:
            released.append(self.thread_id)

    monkeypatch.setattr(reconcile_module, "ThreadLock", _FakeLock)
    return released


# ---------------------------------------------------------------------------
# Distinguishable teardown outcomes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expired_terminal_session_is_deleted_and_confirmed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_reconcile(
        monkeypatch,
        _ThreadReconcileOutcome(status="completed", checkpoint_id="cp-1", failed=False),
    )
    repo = _Repo(current_id="cp-1", current_ts=OLDER_THAN - timedelta(days=1))
    checkpointer = _Checkpointer()

    deleted = await _gc_one_session(
        repo=repo,  # type: ignore[arg-type]
        graph=object(),  # type: ignore[arg-type]
        checkpointer=checkpointer,  # type: ignore[arg-type]
        redis=_Redis(),
        session_meta=_meta(),
        now=NOW,
        older_than=OLDER_THAN,
    )

    assert deleted is True
    assert checkpointer.deleted == [SESSION_ID], "the checkpoint is pruned"
    assert repo.confirms == [SESSION_ID], "the audit stamp is confirmed closed"
    assert repo.stamps[0]["pending"] is True, "the stamp lands BEFORE the delete"


@pytest.mark.asyncio
async def test_generation_moved_session_survives_undeleted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A verdict from an old checkpoint never deletes a NEWER one."""
    _patch_reconcile(
        monkeypatch,
        _ThreadReconcileOutcome(status="completed", checkpoint_id="cp-1", failed=False),
    )
    repo = _Repo(current_id="cp-2-newer", current_ts=NOW - timedelta(minutes=1))
    checkpointer = _Checkpointer()

    deleted = await _gc_one_session(
        repo=repo,  # type: ignore[arg-type]
        graph=object(),  # type: ignore[arg-type]
        checkpointer=checkpointer,  # type: ignore[arg-type]
        redis=_Redis(),
        session_meta=_meta(),
        now=NOW,
        older_than=OLDER_THAN,
    )

    assert deleted is False
    assert checkpointer.deleted == []
    assert repo.stamps == [], "nothing is stamped when the sweep refuses"


@pytest.mark.asyncio
async def test_new_turn_past_retention_boundary_survives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_reconcile(
        monkeypatch,
        _ThreadReconcileOutcome(status="completed", checkpoint_id="cp-1", failed=False),
    )
    repo = _Repo(current_id="cp-1", current_ts=NOW - timedelta(minutes=5))
    checkpointer = _Checkpointer()

    deleted = await _gc_one_session(
        repo=repo,  # type: ignore[arg-type]
        graph=object(),  # type: ignore[arg-type]
        checkpointer=checkpointer,  # type: ignore[arg-type]
        redis=_Redis(),
        session_meta=_meta(),
        now=NOW,
        older_than=OLDER_THAN,
    )

    assert deleted is False
    assert checkpointer.deleted == []


@pytest.mark.asyncio
async def test_failed_catchup_keeps_the_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_reconcile(monkeypatch, _ThreadReconcileOutcome(failed=True))
    repo = _Repo(current_id="cp-1", current_ts=OLDER_THAN - timedelta(days=1))
    checkpointer = _Checkpointer()

    deleted = await _gc_one_session(
        repo=repo,  # type: ignore[arg-type]
        graph=object(),  # type: ignore[arg-type]
        checkpointer=checkpointer,  # type: ignore[arg-type]
        redis=_Redis(),
        session_meta=_meta(),
        now=NOW,
        older_than=OLDER_THAN,
    )

    assert deleted is False
    assert checkpointer.deleted == []
    assert repo.stamps == []


@pytest.mark.asyncio
async def test_pending_retry_with_confirmed_absence_completes_the_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker that died between stamp and delete retries idempotently: the
    checkpoint is already gone, so the no-op catch-up + absent-peek confirms."""
    _patch_reconcile(monkeypatch, _ThreadReconcileOutcome(clean=True, failed=False))
    repo = _Repo(current_id=None, current_ts=None)
    checkpointer = _Checkpointer()

    deleted = await _gc_one_session(
        repo=repo,  # type: ignore[arg-type]
        graph=object(),  # type: ignore[arg-type]
        checkpointer=checkpointer,  # type: ignore[arg-type]
        redis=_Redis(),
        session_meta=_meta(gc_pending=True),
        now=NOW,
        older_than=OLDER_THAN,
    )

    assert deleted is True
    assert checkpointer.deleted == [SESSION_ID]
    assert repo.confirms == [SESSION_ID]


@pytest.mark.asyncio
async def test_locked_candidate_is_never_garbage_collected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live connection owns the thread — the sweep skips it entirely."""
    _patch_reconcile(
        monkeypatch,
        _ThreadReconcileOutcome(status="completed", checkpoint_id="cp-1", failed=False),
    )
    repo = _Repo(
        current_id="cp-1",
        current_ts=OLDER_THAN - timedelta(days=1),
        candidate=_meta(),
    )
    checkpointer = _Checkpointer()
    _make_lock(monkeypatch, acquired=False)

    deleted = await _gc_checkpointed_threads(
        repo=repo,  # type: ignore[arg-type]
        graph=object(),  # type: ignore[arg-type]
        checkpointer=checkpointer,  # type: ignore[arg-type]
        redis=_Redis(),
        now=NOW,
        older_than=OLDER_THAN,
        batch_size=5,
    )

    assert deleted == 0
    assert checkpointer.deleted == []


@pytest.mark.asyncio
async def test_confirm_failure_after_a_successful_delete_ages_into_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deletion succeeded but the confirm failed: the pending record ages into
    a retry whose idempotent delete + confirm self-heals — never a re-delete."""
    _patch_reconcile(
        monkeypatch,
        _ThreadReconcileOutcome(status="completed", checkpoint_id="cp-1", failed=False),
    )
    repo = _Repo(current_id="cp-1", current_ts=OLDER_THAN - timedelta(days=1), confirm_ok=False)
    checkpointer = _Checkpointer()

    deleted = await _gc_one_session(
        repo=repo,  # type: ignore[arg-type]
        graph=object(),  # type: ignore[arg-type]
        checkpointer=checkpointer,  # type: ignore[arg-type]
        redis=_Redis(),
        session_meta=_meta(),
        now=NOW,
        older_than=OLDER_THAN,
    )

    assert deleted is True, "the checkpoint WAS deleted — the sweep reports truth"
    assert checkpointer.deleted == [SESSION_ID]
    assert repo.confirms == [SESSION_ID]
