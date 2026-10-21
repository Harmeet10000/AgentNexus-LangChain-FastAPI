"""Caller-first interface for validated background jobs.

The module contains only producer-side contracts.  Concrete transports are
adapters supplied by the application composition root; importing this module
must not construct a broker client or a database connection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, SkipValidation, ValidationError

from app.shared.jobs.catalog import JobDefinition, get_job, registered_names
from app.shared.jobs.payloads import CeleryTaskPayload

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from typing import Any, Self, TypeGuard


def _payload_error(task_name: str, error: ValidationError) -> Exception:
    """Load legacy dispatch errors only on the invalid path.

    Valid producer imports must not load Celery merely to define the JobBus
    interface.  The compatibility registry remains the owner of these error
    classes, so existing relay/dispatch catches retain their behavior.
    """
    from app.shared.jobs.registry import TaskPayloadValidationError  # noqa: PLC0415

    return TaskPayloadValidationError(task_name, error)


def _unknown_error(task_name: str, names: frozenset[str]) -> Exception:
    """Load the compatibility unknown-name error only when it is needed."""
    from app.shared.jobs.registry import UnregisteredTaskError  # noqa: PLC0415

    return UnregisteredTaskError(task_name, known_names=names)


class EnqueueOptions(BaseModel):
    """Transport-neutral timing options for direct publication.

    Queue and routing are deliberately absent.  The immutable catalog owns
    queue selection; a caller cannot bypass topology by supplying a queue.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    expires: float | None = Field(default=None, ge=0)
    countdown: float | None = Field(default=None, ge=0)


class AggregateRef(BaseModel):
    """Identity attached to a durable outbox entry, not to its payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    aggregate_type: str = Field(min_length=1)
    aggregate_id: str = Field(min_length=1)


class TransactionScope(Protocol):
    """The caller-owned transaction behavior needed by staging adapters.

    This deliberately names the SQL execution capability without importing
    SQLAlchemy into the producer-facing package.  ``AsyncSession`` and
    ``AsyncConnection`` satisfy it structurally; test transactions can expose
    the same narrow method without pretending to be a database session.
    """

    # ``Any`` is confined to the third-party method boundary: SQLAlchemy's
    # overloaded ``execute`` result is deliberately not part of this seam.
    execute: Callable[..., Awaitable[Any]]


def is_transaction_scope(value: object) -> TypeGuard[TransactionScope]:
    """Narrow an injected transaction by its one required capability."""
    return callable(getattr(value, "execute", None))


class TypedJob[T: CeleryTaskPayload](BaseModel):
    """A catalog-owned, validated job ready for an adapter.

    ``TypedJob`` is intentionally not constructible from a raw task name.  Use
    ``JOB_CATALOG[name].prepare(payload)`` (or ``make``) so the immutable
    definition selects both the wire name and payload model.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    # Pydantic would otherwise revalidate an erased catalog definition into a
    # fresh generic instance, losing the identity that proves routing came from
    # the immutable catalog.  The adapter-side identity check remains the
    # runtime trust boundary; this annotation keeps the static payload coupling.
    definition: Annotated[JobDefinition[T], SkipValidation]
    payload: T

    @classmethod
    def from_definition(cls, definition: JobDefinition[T], payload: object) -> Self:
        """Validate ``payload`` against ``definition`` before constructing a job."""
        try:
            validated = definition.payload_model.model_validate(payload)
        except ValidationError as exc:
            raise _payload_error(definition.name, exc) from exc
        return cls(definition=definition, payload=validated)

    @property
    def task_name(self) -> str:
        """Stable wire name selected by the catalog."""
        return self.definition.name

    @property
    def queue(self) -> str:
        """Logical queue role selected by the catalog."""
        return self.definition.queue.value

    def wire_payload(self) -> dict[str, object]:
        """Return the stable JSON-compatible payload sent to an adapter."""
        return cast("dict[str, object]", self.payload.model_dump(mode="json"))

    def wire_json(self) -> str:
        """Return the payload's canonical Pydantic JSON representation."""
        return self.payload.model_dump_json()


class DirectEnqueueReceipt(BaseModel):
    """Receipt for a broker publication; its ID is a Celery task ID."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["direct"] = "direct"
    task_id: str = Field(min_length=1)


class DurableEnqueueReceipt(BaseModel):
    """Receipt for a transaction-staged publication; its ID is an outbox ID."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["durable"] = "durable"
    event_id: str = Field(min_length=1)


class DirectPublicationAdapter(Protocol):
    """Minimum adapter needed to publish a job directly to a broker."""

    async def publish(
        self, job: TypedJob[CeleryTaskPayload], *, options: EnqueueOptions
    ) -> str: ...


class TransactionalStagingAdapter(Protocol):
    """Minimum adapter needed to stage a job in a caller-owned transaction."""

    async def stage(
        self,
        tx: TransactionScope,
        job: TypedJob[CeleryTaskPayload],
        *,
        aggregate: AggregateRef,
    ) -> str: ...


def _validated_job(job: TypedJob[CeleryTaskPayload]) -> TypedJob[CeleryTaskPayload]:
    """Re-check the catalog contract immediately before adapter side effects."""
    definition = get_job(job.task_name)
    if definition is None:
        raise _unknown_error(job.task_name, get_job_names())
    # A caller can only obtain a valid job through a catalog definition.  The
    # identity check also rejects a hand-built definition with the same fields,
    # preventing an adapter from accepting caller-controlled routing metadata.
    if definition is not job.definition:
        raise _unknown_error(job.task_name, frozenset({*get_job_names()}))
    try:
        definition.payload_model.model_validate(job.wire_payload())
    except ValidationError as exc:
        raise _payload_error(job.task_name, exc) from exc
    return job


def get_job_names() -> frozenset[str]:
    """Return catalog names without exposing the catalog's mutable internals."""
    return registered_names()


class TransactionalJobBus:
    """Transaction-bound view of a :class:`JobBus`."""

    def __init__(self, adapter: TransactionalStagingAdapter, tx: TransactionScope) -> None:
        self._adapter = adapter
        self._tx = tx

    async def enqueue(
        self,
        job: TypedJob[CeleryTaskPayload],
        *,
        aggregate: AggregateRef,
    ) -> DurableEnqueueReceipt:
        """Stage a validated job without committing, rolling back, or publishing."""
        validated = _validated_job(job)
        event_id = await self._adapter.stage(self._tx, validated, aggregate=aggregate)
        return DurableEnqueueReceipt(event_id=event_id)


class JobBus:
    """Validated direct and caller-owned transactional enqueue interface."""

    def __init__(
        self,
        direct_adapter: DirectPublicationAdapter,
        transactional_adapter: TransactionalStagingAdapter,
    ) -> None:
        self._direct_adapter = direct_adapter
        self._transactional_adapter = transactional_adapter

    async def enqueue(
        self,
        job: TypedJob[CeleryTaskPayload],
        options: EnqueueOptions | None = None,
    ) -> DirectEnqueueReceipt:
        """Publish a validated job directly and return its broker task ID."""
        validated = _validated_job(job)
        task_id = await self._direct_adapter.publish(
            validated,
            options=options if options is not None else EnqueueOptions(),
        )
        return DirectEnqueueReceipt(task_id=task_id)

    def in_transaction(self, tx: TransactionScope) -> TransactionalJobBus:
        """Bind staging to ``tx``; transaction lifecycle remains caller-owned."""
        return TransactionalJobBus(self._transactional_adapter, tx)


def prepare_job(task_name: str, payload: object) -> TypedJob[CeleryTaskPayload]:
    """Compatibility-friendly helper that still delegates construction to catalog data."""
    definition = get_job(task_name)
    if definition is None:
        raise _unknown_error(task_name, get_job_names())
    return definition.prepare(payload)


def require_direct_receipt(receipt: object, *, operation: str) -> str:
    """Return a broker task id or raise a transport failure before acking.

    Direct and durable receipts are intentionally different types.  A relay
    may only mark an outbox row published after seeing a direct receipt (or the
    legacy adapter's string id); accepting a durable event id here would make
    an unpublished event look delivered.
    """
    task_id: str | None = None
    if isinstance(receipt, DirectEnqueueReceipt):
        if receipt.kind == "direct":
            task_id = receipt.task_id
    elif isinstance(receipt, str):
        task_id = receipt
    if isinstance(task_id, str) and task_id.strip():
        return task_id

    from app.utils import ServiceUnavailableException  # noqa: PLC0415

    error = ServiceUnavailableException("Direct publication returned no valid task id")
    error.add_note(f"operation={operation}")
    error.add_note(f"receipt_type={type(receipt).__name__}")
    raise error


__all__ = [
    "AggregateRef",
    "DirectEnqueueReceipt",
    "DirectPublicationAdapter",
    "DurableEnqueueReceipt",
    "EnqueueOptions",
    "JobBus",
    "TransactionScope",
    "TransactionalJobBus",
    "TransactionalStagingAdapter",
    "TypedJob",
    "is_transaction_scope",
    "prepare_job",
    "require_direct_receipt",
]
