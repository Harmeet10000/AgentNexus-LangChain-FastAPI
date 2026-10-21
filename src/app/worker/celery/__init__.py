"""Canonical worker runtime interface.

The package is deliberately the only place that composes a worker application.
``app.worker.app`` remains a compatibility import for deployments that have
not yet changed their ``-A`` value.
"""

from typing import TYPE_CHECKING

from app.worker.celery.profiles import (
    WORKER_PROFILES,
    WorkerProfile,
    WorkerRole,
    profile_for_role,
)

if TYPE_CHECKING:
    from app.worker.celery.app import celery_app

__all__ = [
    "WORKER_PROFILES",
    "WorkerProfile",
    "WorkerRole",
    "celery_app",
    "profile_for_role",
]


def __getattr__(name: str) -> object:
    """Load the Celery app only when a consumer explicitly requests it."""

    if name == "celery_app":
        from app.worker.celery.app import celery_app  # noqa: PLC0415

        return celery_app
    raise AttributeError(name)
