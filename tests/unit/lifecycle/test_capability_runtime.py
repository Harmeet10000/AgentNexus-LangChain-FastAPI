"""Contract tests for the shared worker capability owner."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field

import pytest

from app.worker.capabilities import owner as capability_runtime

pytestmark = pytest.mark.unit


@dataclass
class FakeLifecycle:
    provisions: int = 0
    releases: int = 0
    loops: list[asyncio.AbstractEventLoop] = field(default_factory=list)

    async def provision(self) -> object:
        self.provisions += 1
        self.loops.append(asyncio.get_running_loop())
        return object()

    async def release(self, _resources: object) -> None:
        self.releases += 1
        self.loops.append(asyncio.get_running_loop())


def test_owner_reuses_one_loop_and_releases_before_idempotent_shutdown() -> None:
    lifecycle = FakeLifecycle()
    owner = capability_runtime.CapabilityOwner(lifecycle)
    loops: list[asyncio.AbstractEventLoop] = []

    assert owner.run(lambda: _capture_loop(loops)) is loops[0]
    assert owner.run(lambda: _capture_loop(loops)) is loops[0]
    owner.shutdown()
    owner.shutdown()

    assert lifecycle.provisions == 1
    assert lifecycle.releases == 1
    assert lifecycle.loops == [loops[0], loops[0]]
    assert loops[0].is_closed()


def test_owner_resets_inherited_state_without_closing_parent_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle = FakeLifecycle()
    owner = capability_runtime.CapabilityOwner(lifecycle)
    parent_loop = owner.run(lambda: _capture_loop([]))
    parent_runner = owner._runner
    assert parent_runner is not None

    monkeypatch.setattr(capability_runtime.os, "getpid", lambda: 42)
    child_loop = owner.run(lambda: _capture_loop([]))
    assert child_loop is not parent_loop
    assert not parent_loop.is_closed()
    owner.shutdown()
    parent_runner.close()


def test_owner_cancels_and_drains_timed_out_task() -> None:
    lifecycle = FakeLifecycle()
    owner = capability_runtime.CapabilityOwner(lifecycle)
    drained = False

    async def long_operation() -> None:
        nonlocal drained
        try:
            await asyncio.sleep(30)
        finally:
            drained = True

    async def timed_operation() -> None:
        async with asyncio.timeout(0.001):
            await long_operation()

    with pytest.raises(TimeoutError):
        owner.run(timed_operation)

    assert drained
    assert owner.run(lambda: _capture_loop([])) is lifecycle.loops[0]
    owner.shutdown()


def test_owner_serializes_concurrent_sync_calls() -> None:
    lifecycle = FakeLifecycle()
    owner = capability_runtime.CapabilityOwner(lifecycle)
    barrier = threading.Barrier(3)
    results: list[object] = []
    errors: list[BaseException] = []

    def invoke() -> None:
        barrier.wait()
        try:
            results.append(owner.run(lambda: _capture_loop([])))
        except BaseException as exc:  # pragma: no cover - assertion below reports failures
            errors.append(exc)

    threads = [threading.Thread(target=invoke) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    owner.shutdown()
    assert errors == []
    assert len(results) == 2
    assert lifecycle.provisions == 1


def test_nested_sync_operations_fail_without_corrupting_owner() -> None:
    lifecycle = FakeLifecycle()
    owner = capability_runtime.CapabilityOwner(lifecycle)

    async def nested() -> None:
        with pytest.raises(capability_runtime.CapabilityUsageError):
            owner.ensure()
        with pytest.raises(capability_runtime.CapabilityUsageError):
            owner.run(lambda: _capture_loop([]))
        competing = asyncio.create_task(owner.reset_in_loop())
        with pytest.raises(capability_runtime.CapabilityUsageError):
            await competing
        await owner.reset_in_loop()
        with pytest.raises(capability_runtime.CapabilityUsageError):
            owner.ensure()

    owner.run(nested)
    assert lifecycle.provisions == 1
    assert lifecycle.releases == 1
    owner.run(lambda: _capture_loop([]))
    assert lifecycle.provisions == 2
    owner.shutdown()


async def _capture_loop(target: list[asyncio.AbstractEventLoop]) -> asyncio.AbstractEventLoop:
    loop = asyncio.get_running_loop()
    target.append(loop)
    await asyncio.sleep(0)
    return loop
