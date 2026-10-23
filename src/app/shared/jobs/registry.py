"""Compatibility facade over the immutable producer-side job catalog."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from celery.exceptions import CeleryError
from pydantic import ValidationError

from app.observability.logging import logger
from app.shared.jobs.catalog import JOB_CATALOG, get_job, payload_models, registered_names

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any, ClassVar

    from app.shared.jobs.payloads import CeleryTaskPayload


class TaskDispatchError(CeleryError):
    """Base for the refusals the typed registry raises before a send."""


class UnregisteredTaskError(TaskDispatchError):
    """A dispatch named a task that has no catalog payload model."""

    def __init__(self, task_name: str, *, known_names: frozenset[str]) -> None:
        self.task_name = task_name
        self.known_names = known_names
        message = (
            f"Celery task {task_name!r} has no registered payload model, so the dispatch was "
            f"refused rather than sent to a name no consumer may answer to. Register a "
            f"CeleryTaskPayload subclass for it in app.shared.jobs.payloads and the shared registry. "
            f"Registered names: {sorted(known_names)}"
        )
        super().__init__(message)


class TaskPayloadValidationError(TaskDispatchError):
    """A dispatched payload did not match the model its catalog entry declares."""

    def __init__(self, task_name: str, validation_error: ValidationError) -> None:
        self.task_name = task_name
        self.validation_error = validation_error
        message = (
            f"Payload for Celery task {task_name!r} does not match its registered model, so the "
            f"dispatch was refused rather than enqueued for a consumer that cannot accept it: "
            f"{validation_error}"
        )
        super().__init__(message)


class CeleryTaskRegistry:
    """Validate-only facade over the immutable job catalog.

    This registry publishes nothing: producers enqueue through
    :class:`app.shared.jobs.bus.JobBus` (or the transactional JobBus seam), and
    this class only validates a payload against its catalog entry. Publication
    methods were removed on purpose — a second send path is a second failure
    mode under incident. ``register`` no longer exists either: allowing a
    producer process to mutate its view of a stable wire contract was the
    split-brain failure this facade removed. Existing reads and validation
    exceptions retain their names and semantics.
    """

    _registry: ClassVar[Mapping[str, type[CeleryTaskPayload]]] = MappingProxyType(
        dict(payload_models())
    )

    @classmethod
    def get(cls, task_name: str) -> type[CeleryTaskPayload] | None:
        """Return the payload model for a known name without importing consumers."""

        definition = get_job(task_name)
        return None if definition is None else definition.payload_model

    @classmethod
    def registered_names(cls) -> frozenset[str]:
        """Return all catalog names as an immutable set."""

        return registered_names()

    @classmethod
    def validate(cls, task_name: str, kwargs: dict[str, Any]) -> CeleryTaskPayload:
        """Validate producer kwargs against the immutable catalog entry."""
        definition = get_job(task_name)
        if definition is None:
            logger.bind(task=task_name, registered=sorted(JOB_CATALOG)).error(
                "Task name is not registered"
            )
            raise UnregisteredTaskError(task_name, known_names=cls.registered_names())
        try:
            return definition.payload_model.model_validate(kwargs)
        except ValidationError as exc:
            logger.bind(task=task_name, errors=exc.errors()).exception(
                "Task payload validation failed"
            )
            raise TaskPayloadValidationError(task_name, exc) from exc


__all__ = [
    "CeleryTaskRegistry",
    "TaskDispatchError",
    "TaskPayloadValidationError",
    "UnregisteredTaskError",
]
