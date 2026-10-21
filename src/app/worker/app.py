"""Compatibility import for the canonical :mod:`app.worker.celery` runtime."""

from app.worker.celery.app import (
    CatalogValidatingBeat,
    CatalogValidatingWorker,
    create_worker_celery_app,
    get_worker_celery_app,
)

celery_app: object

__all__ = [
    "CatalogValidatingBeat",
    "CatalogValidatingWorker",
    "celery_app",
    "create_worker_celery_app",
    "get_worker_celery_app",
]


def __getattr__(name: str) -> object:
    if name == "celery_app":
        return get_worker_celery_app()
    raise AttributeError(name)
