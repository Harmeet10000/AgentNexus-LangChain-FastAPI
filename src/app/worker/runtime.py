"""Persistent async execution for sequential Celery prefork/solo tasks."""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Coroutine
    from typing import Any

_runner: asyncio.Runner | None = None
_pid: int | None = None


def reset_after_fork() -> None:
    """Discard inherited loop references without operating on parent resources."""
    global _runner, _pid  # noqa: PLW0603 — process-owned runtime
    _runner = None
    _pid = os.getpid()


def run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    """Run one coroutine on this process's loop, retaining it between tasks."""
    global _runner  # noqa: PLW0603 — process-owned runtime
    if _pid != os.getpid():
        reset_after_fork()
    if _runner is None:
        _runner = asyncio.Runner()
    return _runner.run(coroutine)


def close() -> None:
    """Close the current child's loop after its resource owners finish cleanup."""
    global _runner  # noqa: PLW0603 — process-owned runtime
    runner = _runner
    _runner = None
    if runner is not None and _pid == os.getpid():
        runner.close()
