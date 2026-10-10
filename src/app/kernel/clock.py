"""Reusable time helpers for cross-feature code.

The project normally uses `datetime.now(UTC)` directly, but some features
need a stable, injectable clock for tests or deterministic workflow logic.
These helpers keep that abstraction in one place without introducing a
heavy framework.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "Clock",
    "FixedClock",
    "utc_now",
    "utc_timestamp",
]


def utc_now() -> datetime:
    """Return the current UTC time."""
    return datetime.now(UTC)


class Clock:
    """Simple time provider that can be replaced in tests or workflows."""

    def __init__(self, now_fn: Callable[[], datetime] | None = None) -> None:
        self._now_fn = now_fn or utc_now

    def now(self) -> datetime:
        return self._now_fn()

    def now_utc(self) -> datetime:
        value = self._now_fn()
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    def timestamp(self) -> float:
        return self.now_utc().timestamp()

    def isoformat(self) -> str:
        return self.now_utc().isoformat()


class FixedClock(Clock):
    """Clock implementation that always returns the same datetime."""

    def __init__(self, value: datetime) -> None:
        self._value = value
        super().__init__(lambda: self._value)


def utc_timestamp() -> float:
    """Return the current Unix timestamp in UTC."""
    return utc_now().timestamp()
