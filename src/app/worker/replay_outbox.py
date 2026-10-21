"""Replay dead-lettered outbox events to Celery."""

import argparse
import asyncio
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, cast

from celery.exceptions import CeleryError
from pydantic import ValidationError
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.connections.celery_producer import (
    DirectPublicationAdapter as CeleryDirectPublicationAdapter,
)
from app.connections.celery_producer import (
    celery_app,
)
from app.connections.postgres import get_database_url
from app.events.model import DeadLetterEvent
from app.observability.logging import logger
from app.shared.jobs import DirectPublicationAdapter as DirectPublicationPort
from app.shared.jobs import EnqueueOptions, prepare_job, require_direct_receipt
from app.utils import ServiceUnavailableException

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.engine.row import RowMapping


async def _fetch_dead_letters(
    session_factory: async_sessionmaker[AsyncSession],
    limit: int,
) -> AsyncGenerator[dict[str, Any], None]:
    async with session_factory() as session:
        rows: Sequence[RowMapping] = (
            (
                await session.execute(
                    statement=select(
                        DeadLetterEvent.original_event_id,
                        DeadLetterEvent.event_type,
                        DeadLetterEvent.payload,
                    )
                    .order_by(DeadLetterEvent.dead_letter_at)
                    .limit(limit),
                )
            )
            .mappings()
            .all()
        )
    for row in rows:
        yield dict(row)


async def replay(
    limit: int = 50,
    *,
    publisher: DirectPublicationPort | None = None,
) -> int:
    database_url = get_database_url()
    engine = create_async_engine(database_url)
    session_factory = async_sessionmaker(engine)
    resolved_publisher = (
        publisher if publisher is not None else CeleryDirectPublicationAdapter(celery_app)
    )
    replayed = 0

    try:
        async for event in _fetch_dead_letters(session_factory, limit=limit):
            try:
                try:
                    job = prepare_job(
                        event["event_type"],
                        cast("dict[str, object]", event["payload"]),
                    )
                    task_id = await resolved_publisher.publish(job, options=EnqueueOptions())
                    require_direct_receipt(task_id, operation="outbox_replay_publish")
                except ValidationError as exc:
                    exc.add_note(f"event_type={event['event_type']}")
                    error = ServiceUnavailableException(
                        "Direct publication returned no valid task id"
                    )
                    raise error from exc
            except (CeleryError, ServiceUnavailableException) as exc:
                exc.add_note(f"event_type={event['event_type']}")
                logger.error("replay_failed", event_type=event["event_type"], error=str(exc))
                continue
            async with session_factory.begin() as session:
                await session.execute(
                    statement=delete(DeadLetterEvent).where(
                        DeadLetterEvent.original_event_id == event["original_event_id"]
                    ),
                )
            replayed += 1

    finally:
        await engine.dispose()
    return replayed


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay dead-lettered outbox events")
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max events to replay (default 50)",
    )
    args = parser.parse_args()
    count = asyncio.run(replay(limit=args.limit))
    logger.info("replay_outbox_complete", count=count)
