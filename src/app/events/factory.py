"""Outbox-relay connection establishment."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from app.connections.postgres import get_database_url
from app.events import OutboxRelay
from app.observability.logging import logger
from app.shared.jobs import JobBus

if TYPE_CHECKING:
    from celery import Celery
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from app.shared.jobs import DirectPublicationAdapter


async def create_outbox_relay(
    publisher: DirectPublicationAdapter | JobBus | Celery,
    session_factory: async_sessionmaker[AsyncSession],
) -> OutboxRelay:
    """Build the relay and run its startup scan.

    New callers inject the direct publication adapter (or JobBus). The Celery
    argument remains accepted as a staged-rollout compatibility wrapper; it is
    converted once here so relay publication has one implementation.
    Task ownership (spawning and cancelling the listener) stays with the caller.
    """
    dsn = get_database_url(flavour="plain")
    if isinstance(publisher, JobBus) or hasattr(publisher, "publish"):
        resolved_publisher = cast("DirectPublicationAdapter | JobBus", publisher)
    else:
        # Keep the legacy raw-Celery call shape at this composition seam only.
        from app.connections.celery_producer import (  # noqa: PLC0415
            DirectPublicationAdapter as CeleryDirectPublicationAdapter,
        )

        resolved_publisher = CeleryDirectPublicationAdapter(cast("Celery", publisher))
    relay = OutboxRelay(
        database_url=dsn,
        publisher=resolved_publisher,
        session_factory=session_factory,
    )
    await relay.run_startup_scan()
    logger.info("Outbox relay started")
    return relay
