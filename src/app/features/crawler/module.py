"""Import-safe composition descriptor for the crawler slice."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.shared.module import Module

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any


def build_consumers() -> Sequence[Any]:
    """Bind crawler-owned Celery consumers only in a worker process."""
    from .adapters.inbound.worker.tasks import crawl_job  # noqa: PLC0415

    return (crawl_job,)


MODULE = Module(name="crawler", consumers=(build_consumers,))
