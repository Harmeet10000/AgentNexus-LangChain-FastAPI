"""Outbox-relay connection establishment."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.connections.postgres import get_database_url
from app.shared.outbox import OutboxRelay
from app.utils import logger

if TYPE_CHECKING:
    from celery import Celery
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


async def create_outbox_relay(
    celery_app: Celery,
    session_factory: async_sessionmaker[AsyncSession],
) -> OutboxRelay:
    """Build the outbox relay against the plain-DSN database and run its startup scan.

    Task ownership (spawning and cancelling the listener) stays with the caller.
    """
    dsn = get_database_url(flavour="plain")
    relay = OutboxRelay(
        database_url=dsn,
        celery_app=celery_app,
        session_factory=session_factory,
    )
    await relay.run_startup_scan()
    logger.info("Outbox relay started")
    return relay
