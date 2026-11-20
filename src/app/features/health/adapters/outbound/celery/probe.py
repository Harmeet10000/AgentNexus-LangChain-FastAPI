"""Celery health probe."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from ....application.dto.result import ComponentCheck

if TYPE_CHECKING:
    from celery import Celery


class CeleryProbe:
    """Check broker reachability using Celery's connection lifecycle."""

    name = "celery"

    def __init__(self, app: Celery | None) -> None:
        self._app = app

    async def check(self) -> ComponentCheck:
        if self._app is None:
            return ComponentCheck.not_configured()
        try:
            start = time.perf_counter()
            connection = self._app.connection()
            try:
                connection.ensure_connection(max_retries=1, timeout=2)
            finally:
                connection.release()
        except (ConnectionRefusedError, TimeoutError, OSError) as exc:
            exc.add_note("component=celery, operation=check_celery")
            return ComponentCheck.unhealthy(str(exc))
        return ComponentCheck.healthy(
            response_time_ms=round((time.perf_counter() - start) * 1000, 2)
        )
