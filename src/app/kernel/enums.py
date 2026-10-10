"""Cross-feature enums that are shared outside one feature boundary.

The project prefers a very small vocabulary here; `ErrorKind` remains the
canonical example. If a second cross-feature enum is needed, keep it in this
module so other slices can import a single, stable source of truth.
"""

from __future__ import annotations

from enum import Enum, StrEnum

__all__ = [
    "LifecycleState",
    "StatusCodeGroup",
]


class LifecycleState(StrEnum):
    """Shared lifecycle states for async or background workflows."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StatusCodeGroup(Enum):
    """High-level classification for HTTP and external status values."""

    INFORMATIONAL = "informational"
    SUCCESS = "success"
    REDIRECTION = "redirection"
    CLIENT_ERROR = "client_error"
    SERVER_ERROR = "server_error"
