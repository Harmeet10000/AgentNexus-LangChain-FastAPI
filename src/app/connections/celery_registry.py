"""Typed Celery task registry.

Provides a ``CeleryTaskRegistry`` that maps task names to Pydantic models
for input validation, and a ``TypedCeleryTask`` base class that validates
kwargs against registered models before execution.

Usage:
    from app.connections.celery_registry import CeleryTaskRegistry, TypedCeleryTask

    registry = CeleryTaskRegistry()
    registry.register("auth.send_verification_email", VerificationEmailPayload)

    @celery_app.task(name="auth.send_verification_email", base=TypedCeleryTask)
    def send_verification_email(self, **kwargs):
        payload = self.validated_payload  # VerificationEmailPayload
        ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from celery import Task
from pydantic import BaseModel, ValidationError

from app.connections import celery_app
from app.utils import logger

if TYPE_CHECKING:
    from typing import Any, ClassVar


class CeleryTaskPayload(BaseModel):
    """Base for all typed Celery task payloads.

    Subclass this for each task to define its typed inputs.
    """

    model_config = {"extra": "forbid", "frozen": True}


class LegacyTaskPayload(CeleryTaskPayload):
    """Fallback payload for untyped tasks during incremental migration.

    Accepts any kwargs — allows legacy tasks to pass through without a
    dedicated Pydantic model while still being registered.
    """

    model_config = {"extra": "allow", "frozen": True}


class CeleryTaskRegistry:
    """Maps task names → Pydantic payload models for validation."""

    _registry: ClassVar[dict[str, type[CeleryTaskPayload]]] = {}

    @classmethod
    def register(cls, task_name: str, payload_model: type[CeleryTaskPayload]) -> None:
        cls._registry[task_name] = payload_model

    @classmethod
    def get(cls, task_name: str) -> type[CeleryTaskPayload] | None:
        return cls._registry.get(task_name)

    @classmethod
    def typed_send(
        cls, task_name: str, kwargs: dict[str, object], **send_task_opts: object
    ) -> object:
        """Validate kwargs against registered model, then send.

        Falls back to LegacyTaskPayload (accepts any kwargs) if task is not registered.
        """
        cls.validate(task_name, kwargs)
        return celery_app.send_task(task_name, kwargs=kwargs, **send_task_opts)

    @classmethod
    def validate(cls, task_name: str, kwargs: dict[str, Any]) -> CeleryTaskPayload:
        """Validate kwargs against registered model, or fall back to LegacyTaskPayload."""
        model = cls._registry.get(task_name)
        if model is None:
            model = LegacyTaskPayload
        try:
            return model.model_validate(kwargs)
        except ValidationError as exc:
            logger.bind(task=task_name, errors=exc.errors()).error("Task payload validation failed")
            raise


class TypedCeleryTask(Task):
    """Base Celery task that validates kwargs against a registered Pydantic model.

    After validation, ``self.validated_payload`` holds the Pydantic model instance.
    """

    abstract = True

    _validated_payload: CeleryTaskPayload | None = None

    @property
    def validated_payload(self) -> CeleryTaskPayload:
        if self._validated_payload is None:
            msg = "validated_payload accessed before task execution"
            raise RuntimeError(msg)
        return self._validated_payload

    def before_start(self, task_id: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        super().before_start(task_id, args, kwargs)
        task_name = self.name or ""
        self._validated_payload = CeleryTaskRegistry.validate(task_name, kwargs)

    def on_success(
        self,
        retval: Any,
        task_id: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        super().on_success(retval, task_id, args, kwargs)
        if self._validated_payload is not None:
            logger.bind(
                task=self.name, task_id=task_id, payload_type=type(self._validated_payload).__name__
            ).info("Typed task completed")
