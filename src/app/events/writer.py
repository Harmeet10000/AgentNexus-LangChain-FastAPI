"""Transactional outbox adapter and compatibility writer.

The adapter deliberately accepts a caller-owned SQLAlchemy transaction handle.
It writes both the row and the transactional notification on that handle, but
never decides whether the surrounding business operation commits or rolls
back. ``with_outbox`` remains as the legacy, raw-payload facade and now uses
the same catalog validation and adapter implementation.
"""

import json
from collections.abc import Mapping
from inspect import isawaitable
from typing import TYPE_CHECKING, overload, override
from uuid import NAMESPACE_URL, uuid4, uuid5

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
from app.shared.jobs.names import DOCUMENTS_INGEST_TERMINAL
from app.shared.jobs.payloads import CeleryTaskPayload

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

OUTBOX_CHANNEL = "outbox_channel"
TERMINAL_EVENT_TYPE = DOCUMENTS_INGEST_TERMINAL


def terminal_event_id(payload: Mapping[str, object]) -> str:
    """Derive the canonical id for a terminal ingestion event.

    Duplicate deliveries of one attempt collide, while a different tenant,
    document version, status, attempt, or task identity remains distinct.
    JSON encoding avoids delimiter-collision bugs in hand-built keys.
    """
    document_version = payload.get("document_version", 1)
    attempt = payload.get("attempt", 0)
    identity = {
        "tenant_id": str(payload.get("tenant_id", "")),
        "document_id": str(payload.get("document_id", "")),
        "document_version": document_version if isinstance(document_version, int) else 1,
        "status": str(payload.get("status", "")),
        "attempt": attempt if isinstance(attempt, int) else 0,
        "task_id": str(payload.get("task_id") or ""),
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return str(uuid5(NAMESPACE_URL, f"agentnexus:ingestion-terminal:{canonical}"))


async def with_terminal_outbox(
    session: "AsyncSession",
    *,
    document_id: str,
    payload: dict[str, object],
) -> str:
    """Stage one deterministic terminal event in the current transaction.

    ``ON CONFLICT DO NOTHING`` makes retries and Celery redeliveries
    idempotent.  The notification is emitted only when this invocation
    inserted the event, and transaction ownership remains with the caller.
    """
    # The aggregate argument is the authoritative document identity. Keep it
    # in the payload too so the row, event id, and consumer cannot disagree.
    event_payload = {**payload, "document_id": document_id}
    job = prepare_job(TERMINAL_EVENT_TYPE, event_payload)
    # Derive the id from the validated/normalized wire model.  This keeps
    # Pydantic coercions (for example JSON-safe integer values) from making the
    # id disagree with the payload eventually consumed by Celery.
    normalized_payload = job.payload.model_dump(mode="json")
    event_id = terminal_event_id(normalized_payload)
    result = await session.execute(
        text(
            """
            INSERT INTO outbox_events
                (id, aggregate_type, aggregate_id, event_type, payload, created_at)
            VALUES
                (:id, 'user_document', :aggregate_id, :event_type,
                 CAST(:payload AS jsonb), now())
            ON CONFLICT (id) DO NOTHING
            RETURNING id
            """
        ),
        {
            "id": event_id,
            "aggregate_id": document_id,
            "event_type": TERMINAL_EVENT_TYPE,
            "payload": job.wire_json(),
        },
    )
    scalar = result.scalar_one_or_none()
    inserted = await scalar if isawaitable(scalar) else scalar
    if inserted is not None:
        await session.execute(
            text("SELECT pg_notify(:channel, :event_id)"),
            {"channel": OUTBOX_CHANNEL, "event_id": event_id},
        )
    return event_id


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
    "TERMINAL_EVENT_TYPE",
    "PostgresOutboxAdapter",
    "PostgresTransactionalStagingAdapter",
    "TransactionalOutboxAdapter",
    "terminal_event_id",
    "with_outbox",
    "with_terminal_outbox",
]
