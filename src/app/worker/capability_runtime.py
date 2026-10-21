"""Process-local async capability lifetimes for prefork Celery workers.

The worker adapters own *what* a capability is and how it is provisioned or
released.  :class:`CapabilityOwner` owns the less visible runtime invariants:
one child process, one persistent event loop, lazy construction, and orderly
cleanup.  Keeping those invariants here prevents each capability from growing
its own subtly different runner implementation.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import threading
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine
    from typing import Any


class CapabilityLifecycle[T](Protocol):
    """The deliberately small seam between an owner and a capability adapter."""

    async def provision(self) -> T:
        """Construct resources on the owner's loop."""

    async def release(self, resources: T) -> None:
        """Release resources on the loop that constructed them."""


class CapabilityUsageError(RuntimeError):
    """A synchronous owner operation was attempted from a running loop."""


class CapabilityOwner[T]:
    """Own one capability's resources and event loop within one child process.

    The interface is intentionally synchronous at task boundaries: Celery
    tasks call :meth:`ensure`, :meth:`run`, and :meth:`shutdown`, while the
    adapter's asynchronous implementation remains on the persistent loop.
    ``reset_in_loop`` is the one async operation needed by capabilities (such
    as a browser) that can be health-reset while a task is already running.

    Synchronous operations are a sequential task-boundary interface: calls
    from different threads serialize around the runner, while calls made from
    any running event loop fail immediately.  A fork reset only discards
    inherited references.  It never attempts to
    close parent-owned resources from the child.  Resources are always
    released before the runner is closed, and a task that escapes through a
    ``BaseException`` (including a soft-time-limit cancellation) is cancelled
    and drained before the next task can use the loop.
    """

    def __init__(self, lifecycle: CapabilityLifecycle[T]) -> None:
        self._lifecycle = lifecycle
        self._pid: int | None = None
        self._runner: asyncio.Runner | None = None
        self._resources: T | None = None
        self._has_resources = False
        self._active_task: asyncio.Task[Any] | None = None
        self._gate = threading.RLock()
        os.register_at_fork(after_in_child=self._after_fork)

    def ensure(self) -> T:
        """Lazily provision and return resources on the owning loop."""
        self._assert_sync_boundary()
        self._reset_if_forked()
        with self._gate:
            if self._has_resources:
                if self._resources is None:
                    message = "Capability owner has an empty resource state"
                    raise AssertionError(message)
                return self._resources

            runner = self._get_runner()
            try:
                resources = runner.run(self._lifecycle.provision())
            except BaseException:
                # Runner.run() drains its own task, but the runner itself must
                # not survive a failed/aborted provision and be reused with a
                # partial resource graph.
                self._close_runner(runner)
                self._runner = None
                raise
            self._resources = resources
            self._has_resources = True
            return resources

    def run[R](self, coroutine_factory: Callable[[], Coroutine[Any, Any, R]]) -> R:
        """Run a coroutine factory on the loop that owns the capability.

        The factory is invoked inside a wrapper coroutine, rather than while
        constructing the task in the caller's context.  This matters for
        factories that inspect the running loop or acquire loop-bound objects
        before their first await.
        """
        self._assert_sync_boundary()
        self._reset_if_forked()
        with self._gate:
            self.ensure()
            runner = self._runner
            if runner is None:
                message = "Capability owner has resources without a runner"
                raise AssertionError(message)
            loop = runner.get_loop()

            async def invoke() -> R:
                return await coroutine_factory()

            task = loop.create_task(invoke())
            self._active_task = task
            try:
                return loop.run_until_complete(task)
            except BaseException:
                task.cancel()
                with contextlib.suppress(BaseException):
                    loop.run_until_complete(asyncio.gather(task, return_exceptions=True))
                raise
            finally:
                if self._active_task is task:
                    self._active_task = None

    async def reset_in_loop(self) -> None:
        """Release resources while retaining the owning runner.

        Call this only from a coroutine already executing on this owner's
        loop.  The next :meth:`ensure` provisions a replacement capability on
        the same loop, which is essential for loop-affine clients.
        """
        runner = self._runner
        if runner is not None:
            try:
                running_loop = asyncio.get_running_loop()
            except RuntimeError as exc:
                message = "Capability reset must run inside the owning event loop"
                raise CapabilityUsageError(message) from exc
            if running_loop is not runner.get_loop():
                message = "Capability reset must run inside the owning event loop"
                raise CapabilityUsageError(message)
            active_task = self._active_task
            if active_task is not None and asyncio.current_task() is not active_task:
                message = "Capability reset cannot race an in-flight owner operation"
                raise CapabilityUsageError(message)
        resources = self._take_resources()
        if resources is not None:
            await self._lifecycle.release(resources)

    def reset_after_fork(self) -> None:
        """Discard inherited child state without touching parent resources."""
        if self._pid != os.getpid():
            # The child may inherit a gate held by a parent thread; never try
            # to acquire that lock before replacing it.
            self._after_fork()
            return
        with self._gate:
            self._after_fork()

    def shutdown(self) -> None:
        """Release resources once, then close their persistent runner.

        Release failures propagate to the adapter, but state is consumed and
        the runner is still closed.  Calling this method again is therefore a
        no-op, making duplicate Celery shutdown signals harmless.
        """
        self._assert_sync_boundary()
        self._reset_if_forked()
        with self._gate:
            resources = self._take_resources()
            runner = self._runner
            self._runner = None
            try:
                if resources is not None and runner is not None:
                    runner.run(self._lifecycle.release(resources))
            finally:
                if runner is not None:
                    self._close_runner(runner)

    def peek(self) -> T | None:
        """Return provisioned resources without provisioning or changing state."""
        self._reset_if_forked()
        return self._resources if self._has_resources else None

    def _get_runner(self) -> asyncio.Runner:
        runner = self._runner
        if runner is None:
            runner = asyncio.Runner()
            self._runner = runner
        return runner

    def _reset_if_forked(self) -> None:
        pid = os.getpid()
        if self._pid != pid:
            self._after_fork()

    @staticmethod
    def _assert_sync_boundary() -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        message = (
            "Capability owner synchronous operations must run at a task boundary; "
            "use an adapter's async in-loop operation while the owner loop is running"
        )
        raise CapabilityUsageError(message)

    def _after_fork(self) -> None:
        """Reset child state and reinitialize a lock inherited from the parent."""
        self._gate = threading.RLock()
        self._resources = None
        self._has_resources = False
        self._active_task = None
        self._runner = None
        self._pid = os.getpid()

    def _take_resources(self) -> T | None:
        resources = self._resources if self._has_resources else None
        self._resources = None
        self._has_resources = False
        return resources

    @staticmethod
    def _close_runner(runner: asyncio.Runner) -> None:
        runner.close()

    def _state(self) -> tuple[int | None, asyncio.Runner | None, T | None]:
        return self._pid, self._runner, self._resources if self._has_resources else None
