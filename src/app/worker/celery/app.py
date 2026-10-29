"""Canonical Celery application composition for workers and beat.

The module does not construct an app at import time. A role must be selected
before Celery imports task modules; the application remains lazy so producer
imports never load task modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast, override

from celery.apps.beat import Beat
from celery.apps.worker import Worker

from app.config import get_settings
from app.connections.celery_producer import create_celery_app
from app.worker.celery.completeness import (
    ensure_beat_catalog_completeness,
    ensure_worker_catalog_completeness,
    ensure_worker_profile_completeness,
)
from app.worker.celery.profiles import profile_for_role
from app.worker.celery.registration import TASK_MODULES
from app.worker.celery.schedules import BEAT_SCHEDULE
from app.worker.celery.task import ResilientTask

if TYPE_CHECKING:
    from celery import Celery

    from app.worker.celery.profiles import WorkerRole

settings = get_settings()


class _AppState:
    active: Celery | None = None
    compatibility: Celery | None = None
    constructed: bool = False


_apps = _AppState()
celery_app: Celery


def create_worker_celery_app(
    role: WorkerRole | str | None = None,
    *,
    beat: bool = False,
) -> Celery:
    """Compose one worker or beat app after its role has been selected.

    ``role`` selects a strict task-module subset. ``None`` selects the complete
    catalog for decorator registration and explicit full-catalog tooling. Beat
    validates the catalog-owned schedule and routes without importing consumer
    task modules.
    A process constructs exactly one worker or beat app; separate deployments
    are the supported way to run multiple roles.
    """
    if beat and role is not None:
        message = "beat cannot be combined with a worker role"
        raise ValueError(message)
    if _apps.constructed:
        message = "one worker or beat app may be constructed per process"
        raise RuntimeError(message)
    profile = profile_for_role(role) if role is not None else None
    app = create_celery_app()
    app.Task = ResilientTask
    app.conf.update(
        # Beat only publishes catalog-owned names/routes and must not import
        # consumer adapters or provision capability resources.
        include=() if beat else (profile.task_modules if profile is not None else TASK_MODULES),
        imports=("app.worker.celery.signals", "app.worker.celery.observability"),
        beat_schedule=BEAT_SCHEDULE,
        beat_schedule_filename=settings.CELERY_BEAT_SCHEDULE_FILE,
        worker_role=profile.role.value if profile is not None else None,
        worker_task_names=tuple(sorted(profile.task_names)) if profile is not None else (),
    )
    app.Worker = CatalogValidatingWorker
    app.Beat = CatalogValidatingBeat
    _apps.active = app
    _apps.constructed = True
    if profile is None and not beat:
        _apps.compatibility = app
    return app


def get_worker_celery_app() -> Celery:
    """Return the selected app, lazily creating a full-catalog app."""
    if _apps.active is not None:
        return _apps.active
    if _apps.compatibility is None:
        _apps.compatibility = create_worker_celery_app()
    return _apps.compatibility


class CatalogValidatingWorker(Worker):
    """Run role or full-catalog completeness after task modules bind."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        # Celery's click path constructs ``app.Worker(...)`` without an
        # ``app=`` kwarg; the Worker class default is None and only the
        # process-current app can fill it — which this app never is when
        # launched through ``worker_main``. Bind the registry's active
        # app so ``self.app.loader`` exists by the time Celery's
        # ``init_worker()`` runs.
        if kwargs.get("app") is None and _apps.active is not None:
            kwargs["app"] = _apps.active
        super().__init__(*args, **kwargs)
        app = cast("Celery", self.app)
        role = app.conf.get("worker_role")
        if role is None:
            ensure_worker_catalog_completeness(app)
            return
        profile = profile_for_role(role)
        ensure_worker_profile_completeness(
            role,
            consumed_queues=(profile.queue,),
            task_names=app.tasks,
            task_modules=app.conf.include or (),
        )


class CatalogValidatingBeat(Beat):
    """Validate beat's schedule/config surface before starting its scheduler."""

    @override
    def run(self) -> None:
        self.init_loader()
        ensure_beat_catalog_completeness(self.app)
        self.set_process_title()
        self.start_scheduler()


def __getattr__(name: str) -> object:
    """Lazily expose the compatibility app without eager task imports."""
    if name == "celery_app":
        return get_worker_celery_app()
    raise AttributeError(name)


__all__ = [
    "CatalogValidatingBeat",
    "CatalogValidatingWorker",
    "celery_app",
    "create_worker_celery_app",
    "get_worker_celery_app",
]
