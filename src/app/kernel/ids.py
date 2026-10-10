"""Helpers for consistent cross-feature identity values.

The project currently uses UUIDs and feature-local keys directly, but when
multiple slices need the same semantics (request IDs, correlation IDs,
customer IDs, invoice IDs) it is useful to centralize the validation and
formatting logic here instead of repeating ad hoc string handling.
"""

from __future__ import annotations

from collections import UserString
from uuid import UUID, uuid4

__all__ = [
    "CorrelationId",
    "RequestId",
    "new_uuid",
    "normalize_uuid",
]


def new_uuid() -> UUID:
    """Generate a new RFC 4122 UUID."""
    return uuid4()


def normalize_uuid(value: str | UUID) -> UUID:
    """Coerce a UUID-like value to a canonical UUID instance."""
    return value if isinstance(value, UUID) else UUID(str(value))


class RequestId(UserString):
    """Typed request identifier used for logs, tracing, and web requests."""

    @classmethod
    def new(cls) -> RequestId:
        return cls(str(new_uuid()))


class CorrelationId(UserString):
    """Typed correlation identifier for propagating trace context across calls."""

    @classmethod
    def new(cls) -> CorrelationId:
        return cls(str(new_uuid()))
