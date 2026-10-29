"""The per-round lease race (§5.5) through ``_stream_round_guarded``.

Pins the two failure modes the reviewer caught in the first cut:
a cancelled lease-waiter is NOT a loss signal (healthy rounds keep
streaming), and a guard that exits abnormally stops its in-flight stream
before run_session cleanup releases the lock.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from app.features.agent_saul.application.service import AgentSaulService

if TYPE_CHECKING:
    from typing import Any


class _CompletingGraph:
    """astream_events that ends after one event — a normal round."""

    async def astream_events(self, *_args: Any, **_kwargs: Any) -> Any:
        yield {"event": "done"}


class _FakeLock:
    def __init__(self) -> None:
        self.lease_lost = asyncio.Event()

    def is_alive(self) -> bool:
        return not self.lease_lost.is_set()


class _HangingGraph:
    """astream_events that hangs until cancelled."""

    def __init__(self) -> None:
        self.cancelled = asyncio.Event()

    async def astream_events(self, *_args: Any, **_kwargs: Any) -> Any:
        try:
            yield {"event": "noop"}
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


class _ExplodingGraph:
    async def astream_events(self, *_args: Any, **_kwargs: Any) -> Any:
        boom = RuntimeError("graph exploded")
        raise boom
        yield {}  # pragma: no cover — makes this an async generator


def _service(graph: Any, sent: list[Any]) -> AgentSaulService:
    """Minimal instance carrying only the state the race guard touches."""

    async def fake_send(ws: Any, payload: Any) -> None:
        _ = ws
        sent.append(payload)

    log = _FakeLog()
    service = object.__new__(AgentSaulService)
    service._graph = graph  # type: ignore[attr-defined] — race-guard unit harness
    service._log = log  # type: ignore[attr-defined]
    service._send_json = fake_send  # type: ignore[method-assign]
    service._map_event_to_frame = staticmethod(  # type: ignore[method-assign]
        lambda **_kwargs: None
    )
    return service


class _FakeBind:
    def warning(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def info(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class _FakeLog:
    def bind(self, **_kwargs: Any) -> _FakeBind:
        return _FakeBind()


async def _run_guarded(service: AgentSaulService, lock: _FakeLock) -> asyncio.Task[bool]:
    return asyncio.create_task(
        service._stream_round_guarded(
            ws=None, lock=lock, config={}, current_input={}, step=0, thread_id="t"
        )
    )


async def test_healthy_round_is_not_lock_lost() -> None:
    graph = _HangingGraph()
    sent: list[Any] = []
    service = _service(graph, sent)
    task = await _run_guarded(service, _FakeLock())
    # The round streams forever; the lease-waiter being cancelled by cleanup
    # must not be mistaken for a loss (reviewer P1: healthy chats stop early).
    await asyncio.sleep(0.05)
    assert not task.done()
    assert not graph.cancelled.is_set()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # Abnormal guard exit stops the in-flight stream (reviewer P1: cancelled
    # chats keep writing).
    await asyncio.sleep(0.05)
    assert graph.cancelled.is_set()
    assert sent == []


async def test_completed_round_under_healthy_lock_returns_false_sends_nothing() -> None:
    """CodeRabbit (round 5): a round that completes while the lock is held
    returns False and never emits LOCK_LOST — the cancelled waiter is not a
    loss signal."""
    sent: list[Any] = []
    service = _service(_CompletingGraph(), sent)
    result = await asyncio.wait_for(
        service._stream_round_guarded(
            ws=None,
            lock=_FakeLock(),
            config={},
            current_input={},
            step=0,
            thread_id="t",
        ),
        timeout=2,
    )
    assert result is False
    assert sent == []


async def test_lease_lost_midround_cancels_stream_and_frames_lock_lost() -> None:
    graph = _HangingGraph()
    sent: list[Any] = []
    service = _service(graph, sent)
    lock = _FakeLock()
    task = await _run_guarded(service, lock)
    await asyncio.sleep(0.05)
    lock.lease_lost.set()  # mid-round loss
    aborted = await asyncio.wait_for(task, timeout=2)
    assert aborted is True
    assert graph.cancelled.is_set()  # the in-flight stream was stopped
    frames = [p for p in sent if isinstance(p, dict)]
    assert any(p.get("code") == "LOCK_LOST" for p in frames)


async def test_stream_failure_propagates_under_healthy_lease() -> None:
    sent: list[Any] = []
    service = _service(_ExplodingGraph(), sent)
    with pytest.raises(RuntimeError):
        await asyncio.wait_for(
            service._stream_round_guarded(
                ws=None,
                lock=_FakeLock(),
                config={},
                current_input={},
                step=0,
                thread_id="t",
            ),
            timeout=2,
        )
    assert sent == []  # a failed round never masquerades as LOCK_LOST
