"""Outbox helper — writes outbox row + pg_notify in the same transaction."""

from typing import TYPE_CHECKING
from uuid import uuid4

from sqlalchemy import text

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


OUTBOX_CHANNEL = "outbox_channel"


async def with_outbox(
    session: "AsyncSession",
    aggregate_type: str,
    aggregate_id: str,
    event_type: str,
    payload: dict[str, object],
) -> str:
    """Write an outbox event and notify the relay in the same transaction.

    pg_notify() is transactional — the notification is only delivered after
    the enclosing transaction commits. Returns the event ID.
    """
    event_id = str(uuid4())
    await session.execute(
        text(
            """
            INSERT INTO outbox_events (id, aggregate_type, aggregate_id, event_type, payload)
            VALUES (:id, :aggregate_type, :aggregate_id, :event_type, :payload)
            """
        ),
        {
            "id": event_id,
            "aggregate_type": aggregate_type,
            "aggregate_id": aggregate_id,
            "event_type": event_type,
            "payload": payload,
        },
    )
    await session.execute(
        text("SELECT pg_notify(:channel, :event_id)"),
        {"channel": OUTBOX_CHANNEL, "event_id": event_id},
    )
    return event_id
