"""Small, import-safe interface for Celery worker configuration."""

from app.worker.celery.profiles import (
    WORKER_PROFILES,
    WorkerProfile,
    WorkerRole,
    profile_for_role,
)

__all__ = [
    "WORKER_PROFILES",
    "WorkerProfile",
    "WorkerRole",
    "profile_for_role",
]
