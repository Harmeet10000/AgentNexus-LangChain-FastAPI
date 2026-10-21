"""Transactional outbox adapter and compatibility writer.

The adapter deliberately accepts a caller-owned SQLAlchemy transaction handle.
It writes both the row and the transactional notification on that handle, but
never decides whether the surrounding business operation commits or rolls
back. ``with_outbox`` remains as the legacy, raw-payload facade and now uses
the same catalog validation and adapter implementation.
"""

from typing import TYPE_CHECKING, overload, override
from uuid import uuid4

from sqlalchemy import func, insert, text

from app.events.model import OutboxEvent
from app.shared.jobs.bus import (
    AggregateRef,
    TransactionalStagingAdapter,
    TransactionScope,
    TypedJob,
    _validated_job,
    is_transaction_scope,
    prepare_job,
)
from app.shared.jobs.payloads import CeleryTaskPayload

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

OUTBOX_CHANNEL = "outbox_channel"


class PostgresOutboxAdapter(TransactionalStagingAdapter):
    """Stage catalog-owned jobs in a caller-owned PostgreSQL transaction.

    ``tx`` is intentionally supplied per call instead of being constructed or
    retained by the adapter. A SQLAlchemy ``AsyncSession`` and an
    ``AsyncConnection`` both expose the ``execute`` operation needed here, so
    the adapter only relies on that narrow transactional surface. In
    particular, it does not commit, roll back, close, or dispose anything;
    transaction ownership stays with the feature repository/service.
    """

    @override
    async def stage(
        self,
        tx: TransactionScope,
        job: TypedJob[CeleryTaskPayload],
        *,
        aggregate: AggregateRef,
    ) -> str:
        """Insert one outbox event and notify the relay transactionally.

        Validation happens before generating an id or issuing SQL. SQL errors
        intentionally propagate unchanged so the transaction owner can apply
        its normal rollback/error policy.
        """
        validated = _validated_job(job)
        event_id = str(uuid4())

        await tx.execute(
            insert(OutboxEvent).values(
                id=event_id,
                aggregate_type=aggregate.aggregate_type,
                aggregate_id=aggregate.aggregate_id,
                event_type=validated.task_name,
                payload=validated.wire_payload(),
                # Database-generated: keeps relay ordering on one clock even
                # when application hosts disagree about the time.
                created_at=func.now(),
            )
        )
        await tx.execute(
            text("SELECT pg_notify(:channel, :event_id)"),
            {"channel": OUTBOX_CHANNEL, "event_id": event_id},
        )
        return event_id


# Descriptive aliases keep the adapter discoverable for composition roots that
# name the seam after its storage technology or its JobBus role.
PostgresTransactionalStagingAdapter = PostgresOutboxAdapter
TransactionalOutboxAdapter = PostgresOutboxAdapter


@overload
async def with_outbox(
    session: "AsyncSession",
    aggregate_type: str,
    aggregate_id: str,
    event_type: str,
    payload: dict[str, object],
) -> str: ...


@overload
async def with_outbox(
    session: TransactionScope,
    aggregate_type: str,
    aggregate_id: str,
    event_type: str,
    payload: dict[str, object],
) -> str: ...


async def with_outbox(
    session: "TransactionScope | AsyncSession",
    aggregate_type: str,
    aggregate_id: str,
    event_type: str,
    payload: dict[str, object],
) -> str:
    """Compatibility facade over :class:`PostgresOutboxAdapter`.

    The historical raw task-name/payload signature is preserved for existing
    document and auth call sites. The immutable job catalog now owns the event
    name and payload validation, so invalid or unknown calls fail before
    either SQL statement is attempted.
    """
    if not is_transaction_scope(session):
        message = "Outbox transaction must expose async execute"
        raise TypeError(message)
    job = prepare_job(event_type, payload)
    aggregate = AggregateRef(aggregate_type=aggregate_type, aggregate_id=aggregate_id)
    return await PostgresOutboxAdapter().stage(session, job, aggregate=aggregate)


__all__ = [
    "OUTBOX_CHANNEL",
    "PostgresOutboxAdapter",
    "PostgresTransactionalStagingAdapter",
    "TransactionalOutboxAdapter",
    "with_outbox",
]
