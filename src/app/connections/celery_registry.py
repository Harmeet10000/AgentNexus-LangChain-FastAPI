"""Deprecated shim — use `app.connections.celery` instead.

This file remains for one release so `from app.connections.celery_registry import ...`
keeps working. New code must import from `app.connections.celery` directly.
"""

from app.connections.celery import (
    CeleryTaskPayload,
    CeleryTaskRegistry,
    NoKwargsPayload,
    TaskDispatchError,
    TaskPayloadValidationError,
    TypedCeleryTask,
    UnregisteredTaskError,
)

__all__ = [
    "CeleryTaskPayload",
    "CeleryTaskRegistry",
    "NoKwargsPayload",
    "TaskDispatchError",
    "TaskPayloadValidationError",
    "TypedCeleryTask",
    "UnregisteredTaskError",
]
