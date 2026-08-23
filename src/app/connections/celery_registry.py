"""Typed Celery task registry.

Maps task names to Pydantic models so a dispatched payload is checked against the
model its consumer declares — at dispatch time, in the dispatching process,
before anything reaches a broker. ``TypedCeleryTask`` applies the same check on
the consuming side and exposes the parsed model to the task body.

Two defects this shape replaces, both of which produced work that silently never
happened rather than an error anyone could see:

* a name with no registered model used to fall through to a permissive model that
  accepts any kwargs, logging a warning and sending anyway. A typo, or a rename
  that missed a producer, therefore reached a broker as a well-formed message
  addressed to nobody, and Celery discards an unknown name without complaint.
  An unregistered name is now a refusal that names the task.
* a payload missing a parameter the consumer requires used to enqueue cleanly and
  fail in the worker, once per retry, with a traceback that named the worker
  rather than the producer. It is now refused at dispatch, naming the task.

Both refusals are ``CeleryError`` subclasses, and that choice is load-bearing:
the outbox relay's publish path catches ``CeleryError`` to mark an event failed
and retry it toward the dead-letter table. A Pydantic ``ValidationError`` escapes
that catch into the relay's outer blanket handler, which logs a warning and drops
the event — reintroducing, one layer up, the exact invisibility this module
exists to remove. The original ``ValidationError`` is preserved as ``__cause__``
and on the raised error, so no detail is lost.

Usage:
    from app.connections.celery_registry import CeleryTaskRegistry, TypedCeleryTask
    from app.connections.celery_task_names import SEND_VERIFICATION_EMAIL

    CeleryTaskRegistry.register(SEND_VERIFICATION_EMAIL, VerificationEmailPayload)

    @celery_app.task(name=SEND_VERIFICATION_EMAIL, base=TypedCeleryTask)
    def send_verification_email(self, **kwargs):
        payload = self.validated_payload  # VerificationEmailPayload
        ...
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, override

from celery import Task
from celery.exceptions import CeleryError
from pydantic import BaseModel, ValidationError

from app.connections.celery import celery_app
from app.connections.celery_task_names import TASK_DECLARING_MODULES
from app.utils import logger

if TYPE_CHECKING:
    from typing import Any, ClassVar


class CeleryTaskPayload(BaseModel):
    """Base for all typed Celery task payloads.

    Subclass this for each task to define its typed inputs.
    """

    model_config = {"extra": "forbid", "frozen": True}


class NoKwargsPayload(CeleryTaskPayload):
    """Payload for a task that takes no keyword arguments.

    Scheduled jobs are dispatched by the scheduler with an empty payload, so
    their contract is "nothing, and nothing extra" — which ``extra="forbid"`` on
    a field-less model states exactly. Registering this is not a formality: it is
    what distinguishes "this task genuinely accepts no arguments" from "nobody
    ever declared what this task accepts", and only the first should dispatch.
    """


class TaskDispatchError(CeleryError):
    """Base for the refusals the typed registry raises before a send.

    Deriving from ``CeleryError`` puts these inside the outbox relay's existing
    narrow catch, so a refused dispatch is recorded as a failed event and retried
    toward the dead-letter table instead of vanishing into a blanket handler.
    """


class UnregisteredTaskError(TaskDispatchError):
    """A dispatch named a task that has no registered payload model."""

    def __init__(self, task_name: str, *, known_names: frozenset[str]) -> None:
        self.task_name = task_name
        self.known_names = known_names
        message = (
            f"Celery task {task_name!r} has no registered payload model, so the dispatch was "
            f"refused rather than sent to a name no consumer may answer to. Register a "
            f"CeleryTaskPayload subclass for it in the module that declares it. "
            f"Registered names: {sorted(known_names)}"
        )
        super().__init__(message)


class TaskPayloadValidationError(TaskDispatchError):
    """A dispatched payload did not match the model its task declares."""

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
    """Maps task names → Pydantic payload models for validation."""

    _registry: ClassVar[dict[str, type[CeleryTaskPayload]]] = {}

    @classmethod
    def register(cls, task_name: str, payload_model: type[CeleryTaskPayload]) -> None:
        cls._registry[task_name] = payload_model

    @classmethod
    def get(cls, task_name: str) -> type[CeleryTaskPayload] | None:
        return cls._registry.get(task_name)

    @classmethod
    def registered_names(cls) -> frozenset[str]:
        """Names that currently hold a payload model in this process."""
        return frozenset(cls._registry)

    @classmethod
    def ensure_declared_module_imported(cls, task_name: str) -> None:
        """Import the module that declares ``task_name``, if it is not loaded yet.

        Registration happens as a side effect of importing the declaring module,
        and a dispatching process has no reason to have imported it: the API
        process runs the relay without ever touching the task package. Skipping
        this step is what made the whole typed contract inert — every name looked
        unregistered, so every payload was validated against a permissive model.

        Only the one module that declares the requested name is imported.
        Importing the full list instead costs fourteen seconds on a cold process
        because the ingestion modules pull the document-converter stack.
        """
        if task_name in cls._registry:
            return
        module = TASK_DECLARING_MODULES.get(task_name)
        if module is None:
            return
        import_module(module)

    @classmethod
    def typed_send(
        cls, task_name: str, kwargs: dict[str, object], **send_task_opts: object
    ) -> object:
        """Validate kwargs against the task's registered model, then send.

        An unregistered name or a mismatched payload raises before the send, so a
        refused dispatch never reaches a broker.
        """
        cls.ensure_declared_module_imported(task_name)
        cls.validate(task_name, kwargs)
        return celery_app.send_task(task_name, kwargs=kwargs, **send_task_opts)

    @classmethod
    def validate(cls, task_name: str, kwargs: dict[str, Any]) -> CeleryTaskPayload:
        """Parse kwargs with the task's registered model.

        Refuses an unregistered name. The predecessor substituted a permissive
        model here and logged a warning, which meant a misaddressed dispatch was
        indistinguishable from a correct one in every place anybody looked.
        """
        model = cls._registry.get(task_name)
        if model is None:
            logger.bind(task=task_name, registered=sorted(cls._registry)).error(
                "Task name is not registered"
            )
            raise UnregisteredTaskError(task_name, known_names=cls.registered_names())
        try:
            return model.model_validate(kwargs)
        except ValidationError as exc:
            logger.bind(task=task_name, errors=exc.errors()).error("Task payload validation failed")
            raise TaskPayloadValidationError(task_name, exc) from exc


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

    @override
    def before_start(self, task_id: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        super().before_start(task_id, args, kwargs)
        task_name = self.name or ""
        self._validated_payload = CeleryTaskRegistry.validate(task_name, kwargs)

    @override
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
