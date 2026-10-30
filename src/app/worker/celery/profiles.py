"""Immutable worker role profiles.

Queue selection, process shape and task subsets belong together.  Keeping them
in one immutable value prevents a deployment from accidentally combining the
ingestion queue with the default worker's concurrency or consuming the DLQ.
"""

from __future__ import annotations

from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, override

from pydantic import BaseModel, ConfigDict

from app.config import get_settings
from app.shared.jobs.catalog import JOB_CATALOG, JobQueue

if TYPE_CHECKING:
    from typing import Final


class WorkerRole(StrEnum):
    """The supported worker pools; beat is a separate process, not a worker."""

    DEFAULT = "default"
    INGESTION = "ingestion"
    CRAWLER = "crawler"

    # Lowercase aliases keep configuration-oriented callers ergonomic while
    # the canonical enum members remain conventional and discoverable.
    default = DEFAULT
    ingestion = INGESTION
    crawler = CRAWLER


class WorkerProfile(BaseModel):
    """The complete command-level contract for one worker pool."""

    model_config = ConfigDict(frozen=True)

    role: WorkerRole
    queue: str
    concurrency: int
    pool: str
    max_tasks_per_child: int
    max_memory_per_child: int
    task_modules: tuple[str, ...]
    task_names: frozenset[str]

    @property
    def task_module_subset(self) -> tuple[str, ...]:
        """Compatibility spelling for the role's explicit module subset."""

        return self.task_modules

    @override
    def model_post_init(self, __context: object) -> None:
        if not self.queue:
            message = "worker profiles require a queue"
            raise ValueError(message)
        if self.pool != "prefork":
            message = "worker profiles must use the prefork pool"
            raise ValueError(message)
        if self.concurrency < 1:
            message = "worker concurrency must be positive"
            raise ValueError(message)
        if self.max_tasks_per_child < 1:
            message = "max_tasks_per_child must be positive"
            raise ValueError(message)
        if self.max_memory_per_child < 1:
            message = "max_memory_per_child must be positive"
            raise ValueError(message)
        if not self.task_modules:
            message = "worker profiles require at least one task module"
            raise ValueError(message)
        if not self.task_names:
            message = "worker profiles require at least one task name"
            raise ValueError(message)


def _profile(
    role: WorkerRole,
    queue: str,
    *,
    concurrency: int,
    max_tasks_per_child: int,
    max_memory_per_child: int,
    queue_role: JobQueue,
    dead_letter_queue: str,
) -> WorkerProfile:
    if queue == dead_letter_queue:
        message = f"worker role {role.value!r} cannot consume the dead-letter queue"
        raise ValueError(message)
    entries = {
        name: definition
        for name, definition in JOB_CATALOG.items()
        if definition.queue is queue_role
    }
    return WorkerProfile(
        role=role,
        queue=queue,
        concurrency=concurrency,
        pool="prefork",
        max_tasks_per_child=max_tasks_per_child,
        max_memory_per_child=max_memory_per_child,
        task_modules=tuple(dict.fromkeys(d.declaring_module for d in entries.values())),
        task_names=frozenset(entries),
    )


def _build_profiles() -> dict[WorkerRole, WorkerProfile]:
    settings = get_settings()
    return {
        WorkerRole.DEFAULT: _profile(
            WorkerRole.DEFAULT,
            settings.CELERY_DEFAULT_QUEUE,
            concurrency=settings.CELERY_DEFAULT_CONCURRENCY,
            max_tasks_per_child=settings.CELERY_DEFAULT_MAX_TASKS_PER_CHILD,
            max_memory_per_child=settings.CELERY_DEFAULT_MAX_MEMORY_PER_CHILD,
            queue_role=JobQueue.DEFAULT,
            dead_letter_queue=settings.CELERY_DEAD_LETTER_QUEUE,
        ),
        WorkerRole.INGESTION: _profile(
            WorkerRole.INGESTION,
            settings.CELERY_INGESTION_QUEUE,
            concurrency=settings.CELERY_INGESTION_CONCURRENCY,
            max_tasks_per_child=settings.CELERY_INGESTION_MAX_TASKS_PER_CHILD,
            max_memory_per_child=settings.CELERY_INGESTION_MAX_MEMORY_PER_CHILD,
            queue_role=JobQueue.INGESTION,
            dead_letter_queue=settings.CELERY_DEAD_LETTER_QUEUE,
        ),
        WorkerRole.CRAWLER: _profile(
            WorkerRole.CRAWLER,
            settings.CELERY_CRAWLER_QUEUE,
            concurrency=settings.CELERY_CRAWLER_CONCURRENCY,
            max_tasks_per_child=settings.CELERY_CRAWLER_MAX_TASKS_PER_CHILD,
            max_memory_per_child=settings.CELERY_CRAWLER_MAX_MEMORY_PER_CHILD,
            queue_role=JobQueue.CRAWLER,
            dead_letter_queue=settings.CELERY_DEAD_LETTER_QUEUE,
        ),
    }


WORKER_PROFILES: Final = MappingProxyType(_build_profiles())
PROFILES: Final = WORKER_PROFILES


def profile_for_role(role: WorkerRole | str) -> WorkerProfile:
    """Resolve and validate a role supplied by a CLI or deployment."""

    try:
        resolved = role if isinstance(role, WorkerRole) else WorkerRole(role)
    except ValueError as exc:
        supported = ", ".join(item.value for item in WorkerRole)
        message = f"unknown worker role {role!r}; choose one of {supported}"
        raise ValueError(message) from exc
    return WORKER_PROFILES[resolved]


get_worker_profile = profile_for_role

__all__ = [
    "PROFILES",
    "WORKER_PROFILES",
    "WorkerProfile",
    "WorkerRole",
    "get_worker_profile",
    "profile_for_role",
]
