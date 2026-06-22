"""Replay dead-lettered outbox events to Celery."""

import argparse
from collections.abc import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.connections.celery import celery_app
from app.connections.postgres import get_database_url
from app.utils import logger


async def _fetch_dead_letters(
    session_factory: async_sessionmaker[AsyncSession],
    limit: int,
) -> AsyncGenerator[dict, None]:
    async with session_factory() as session:
        rows = (
            await session.execute(
                text(
                    """
                    SELECT original_event_id, event_type, payload
                    FROM dead_letter_events
                    ORDER BY dead_letter_at
                    LIMIT :limit
                    """
                ),
                {"limit": limit},
            )
        ).mappings().all()
        for row in rows:
            yield dict(row)


async def replay(limit: int = 50) -> int:
    database_url = get_database_url()
    engine = create_async_engine(database_url)
    session_factory = async_sessionmaker(engine)
    replayed = 0

    async for event in _fetch_dead_letters(session_factory, limit=limit):
        try:
            celery_app.send_task(event["event_type"], kwargs=event["payload"])
        except Exception as exc:  # noqa: BLE001
            logger.error("replay_failed", event_type=event["event_type"], error=str(exc))
            continue
        async with session_factory.begin() as session:
            await session.execute(
                text("DELETE FROM dead_letter_events WHERE original_event_id = :id"),
                {"id": event["original_event_id"]},
            )
        replayed += 1

    await engine.dispose()
    return replayed


async def main() -> None:
    parser = argparse.ArgumentParser(description="Replay dead-lettered outbox events")
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max events to replay (default 50)",
    )
    args = parser.parse_args()
    count = await replay(limit=args.limit)
    logger.info("replay_outbox_complete", count=count)
