"""Import-safe composition descriptor for the search slice."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.shared.module import Module

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any


def build_consumers() -> Sequence[Any]:
    """Bind search-owned Celery consumers only in a worker process."""
    from .adapters.inbound.worker.tasks import search_ingest  # noqa: PLC0415

    return (search_ingest,)


MODULE = Module(name="search", consumers=(build_consumers,))
