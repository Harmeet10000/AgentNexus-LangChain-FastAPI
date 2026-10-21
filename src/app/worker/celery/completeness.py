"""Role and deployment completeness checks for the canonical runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.worker.celery.profiles import profile_for_role
from app.worker.completeness import (
    CatalogCompletenessError,
    JobCatalogDrift,
    QueueCoverageDrift,
    check_beat_catalog_completeness,
    check_deployment_queue_coverage,
    check_job_catalog_completeness,
    check_worker_catalog_completeness,
    ensure_beat_catalog_completeness,
    ensure_worker_catalog_completeness,
)


def _application_values(values: Iterable[str]) -> frozenset[str]:
    """Drop Celery framework registrations, retaining every app value."""
    return frozenset(value for value in values if not value.startswith("celery."))


if TYPE_CHECKING:
    from collections.abc import Iterable

    from app.worker.celery.profiles import WorkerProfile, WorkerRole


@dataclass(frozen=True, slots=True)
class WorkerProfileDrift:
    """Mismatch between a role profile and the jobs a worker advertises."""

    role: WorkerRole
    expected_queue: str
    consumed_queues: tuple[str, ...]
    missing_task_names: tuple[str, ...] = ()
    unexpected_task_names: tuple[str, ...] = ()
    missing_task_modules: tuple[str, ...] = ()
    unexpected_task_modules: tuple[str, ...] = ()

    @property
    def errors(self) -> tuple[str, ...]:
        values: list[str] = []
        if self.consumed_queues != (self.expected_queue,):
            values.append(
                f"consumed_queues: expected {self.expected_queue}, got {self.consumed_queues}"
            )
        values.extend(f"missing_task_names: {name}" for name in self.missing_task_names)
        values.extend(f"unexpected_task_names: {name}" for name in self.unexpected_task_names)
        values.extend(f"missing_task_modules: {name}" for name in self.missing_task_modules)
        values.extend(f"unexpected_task_modules: {name}" for name in self.unexpected_task_modules)
        return tuple(values)

    @property
    def ok(self) -> bool:
        return not self.errors


def check_worker_profile_completeness(
    role: WorkerRole | str,
    *,
    consumed_queues: Iterable[str] = (),
    task_names: Iterable[str] = (),
    task_modules: Iterable[str] = (),
) -> WorkerProfileDrift:
    """Validate one worker's task subset and queue selection without a broker."""

    profile: WorkerProfile = profile_for_role(role)
    expected_modules = frozenset(profile.task_modules)
    expected_names = profile.task_names
    # Celery installs canvas/backend tasks and ``celery.app.builtins`` on every
    # app. They are framework-owned and intentionally absent from a role
    # profile; application names/modules remain strict so an accidental app
    # import cannot broaden a worker silently.
    actual_names = _application_values(task_names)
    actual_modules = _application_values(task_modules)
    return WorkerProfileDrift(
        role=profile.role,
        expected_queue=profile.queue,
        consumed_queues=tuple(dict.fromkeys(consumed_queues)),
        missing_task_names=tuple(sorted(expected_names - actual_names)),
        unexpected_task_names=tuple(sorted(actual_names - expected_names)),
        missing_task_modules=tuple(sorted(expected_modules - actual_modules)),
        unexpected_task_modules=tuple(sorted(actual_modules - expected_modules)),
    )


def ensure_worker_profile_completeness(
    role: WorkerRole | str,
    *,
    consumed_queues: Iterable[str] = (),
    task_names: Iterable[str] = (),
    task_modules: Iterable[str] = (),
) -> None:
    report = check_worker_profile_completeness(
        role,
        consumed_queues=consumed_queues,
        task_names=task_names,
        task_modules=task_modules,
    )
    if not report.ok:
        message = f"{report.role.value} worker profile is incomplete: {report.errors}"
        raise RuntimeError(message)


__all__ = [
    "CatalogCompletenessError",
    "JobCatalogDrift",
    "QueueCoverageDrift",
    "WorkerProfileDrift",
    "check_beat_catalog_completeness",
    "check_deployment_queue_coverage",
    "check_job_catalog_completeness",
    "check_worker_catalog_completeness",
    "check_worker_profile_completeness",
    "ensure_beat_catalog_completeness",
    "ensure_worker_catalog_completeness",
    "ensure_worker_profile_completeness",
]
