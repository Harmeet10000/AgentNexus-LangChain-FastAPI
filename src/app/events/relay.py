"""Transactional outbox relay using PostgreSQL NOTIFY/LISTEN."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, cast

import asyncpg
import asyncpg_listen
from asyncpg.exceptions import PostgresError
from celery.exceptions import CeleryError
from sqlalchemy import delete, func, select, text, update

from app.events.model import OutboxEvent
from app.observability.logging import logger
from app.utils import ServiceUnavailableException

if TYPE_CHECKING:
    from typing import Final

    from celery import Celery
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

_MAX_RETRIES: Final[int] = 5


class OutboxRelay:
    """Listens for outbox events and publishes them to Celery.

    Startup scan (one-shot): catches events created while relay was offline.
    Listen loop (continuous): subscribes to outbox_channel via asyncpg-listen.
    """

    def __init__(
        self,
        database_url: str,
        celery_app: Celery,
        *,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        """Store the relay's collaborators.

        Args:
            database_url: A plain libpq DSN, as returned by
                `get_database_url(flavour="plain")`. `asyncpg.connect` cannot parse a
                SQLAlchemy dialect scheme, so the URL is used verbatim rather than
                repaired here.
            celery_app: Celery application used to publish events.
            session_factory: Async session factory for the outbox tables.
        """
        self._database_url = database_url
        self._celery_app = celery_app
        self._session_factory = session_factory
        self._accepting_notifications = True
        self._in_flight = 0
        self._drained = asyncio.Event()
        self._drained.set()

    async def run_startup_scan(self) -> None:
        """One-time scan for unpublished events created while relay was offline."""
        try:
            async with self._session_factory() as session:
                rows = (
                    (
                        await session.execute(
                            select(
                                OutboxEvent.id,
                                OutboxEvent.event_type,
                                OutboxEvent.payload,
                                OutboxEvent.publish_attempts,
                            )
                            .where(
                                OutboxEvent.published_at.is_(None),
                                OutboxEvent.publish_attempts < _MAX_RETRIES,
                            )
                            .order_by(OutboxEvent.created_at)
                            .limit(100)
                            .with_for_update(skip_locked=True),
                        )
                    )
                    .mappings()
                    .all()
                )

                if rows:
                    logger.bind(found=len(rows)).info("outbox_startup_scan")
                    for row in rows:
                        await self._publish(dict(row), session=session)
                else:
                    logger.bind(found=0).info("outbox_startup_scan")
        except Exception as exc:  # noqa: BLE001 — best-effort scan, any failure is logged and skipped
            logger.bind(error=str(exc)).warning("outbox_startup_scan_skipped")

    async def run_listener(self) -> None:
        """Long-running listen loop. Subscribe to outbox_channel, handle notifications."""
        try:
            listener = asyncpg_listen.NotificationListener(
                connect=lambda: asyncpg.connect(dsn=self._database_url),
            )
            await listener.run(
                handler_per_channel={"outbox_channel": self._handle_notification},
                policy=asyncpg_listen.ListenPolicy.ALL,
                notification_timeout=asyncpg_listen.NO_TIMEOUT,
            )
        except Exception as exc:  # noqa: BLE001 — listener is long-running; log and re-enter loop
            logger.bind(error=str(exc)).warning("outbox_listener_stopped")

    async def _handle_notification(
        self,
        notification: asyncpg_listen.Notification | asyncpg_listen.Timeout,
    ) -> None:
        if isinstance(notification, asyncpg_listen.Timeout):
            return
        if not self._accepting_notifications:
            return
        event_id = notification.payload
        if not event_id:
            return

        self._in_flight += 1
        self._drained.clear()
        try:
            async with self._session_factory() as session:
                result = (
                    (
                        await session.execute(
                            select(
                                OutboxEvent.id,
                                OutboxEvent.event_type,
                                OutboxEvent.payload,
                                OutboxEvent.publish_attempts,
                            )
                            .where(
                                OutboxEvent.id == event_id,
                                OutboxEvent.published_at.is_(None),
                                OutboxEvent.publish_attempts < _MAX_RETRIES,
                            )
                            .with_for_update(skip_locked=True),
                        )
                    )
                    .mappings()
                    .one_or_none()
                )
                if result is None:
                    return
                await self._publish(dict(result), session=session)
        finally:
            self._in_flight -= 1
            if self._in_flight == 0:
                self._drained.set()

    async def drain(self, *, timeout_seconds: float = 5.0) -> bool:
        """Stop accepting notifications and wait for active publishes to finish."""
        self._accepting_notifications = False
        try:
            await asyncio.wait_for(self._drained.wait(), timeout=timeout_seconds)
        except TimeoutError:
            logger.bind(in_flight=self._in_flight, timeout_seconds=timeout_seconds).warning(
                "outbox_drain_timed_out"
            )
            return False
        return True

    async def _publish(
        self,
        row: dict[str, object],
        session: AsyncSession,
    ) -> None:
        event_id = str(row["id"])
        event_type = str(row["event_type"])
        payload = cast("dict[str, object]", row["payload"])
        try:
            # Local import: connections/__init__ reaches back into app.events.
            from app.connections import celery_producer  # noqa: PLC0415

            await celery_producer.dispatch(self._celery_app, event_type, payload)
            await self._mark_published(event_id, session=session)
            logger.bind(event_id=event_id, event_type=event_type).info("outbox_published")
        except (CeleryError, PostgresError, ServiceUnavailableException) as exc:
            exc.add_note(f"event_id={event_id}, event_type={event_type}")
            await self._mark_failed(event_id, str(exc), session=session)
            logger.bind(event_id=event_id, event_type=event_type, error=str(exc)).exception(
                "outbox_publish_failed"
            )

    @staticmethod
    async def _mark_published(
        event_id: str,
        session: AsyncSession,
    ) -> None:
        await session.execute(
            update(OutboxEvent).where(OutboxEvent.id == event_id).values(published_at=func.now()),
        )
        await session.commit()

    @staticmethod
    async def _mark_failed(
        event_id: str,
        error: str,
        session: AsyncSession,
    ) -> None:
        row = (
            (
                await session.execute(
                    select(OutboxEvent.publish_attempts)
                    .where(OutboxEvent.id == event_id)
                    .with_for_update(),
                )
            )
            .mappings()
            .one_or_none()
        )
        if row is None:
            return
        attempts = row["publish_attempts"] + 1
        if attempts >= _MAX_RETRIES:
            # Kept as explicit SQL: INSERT ... SELECT across the two outbox
            # tables with gen_random_uuid() is clearer spelled out than composed.
            await session.execute(
                text(
                    """
                    INSERT INTO dead_letter_events
                        (id, original_event_id, aggregate_type, aggregate_id, event_type,
                         payload, created_at, dead_letter_at, last_error)
                    SELECT gen_random_uuid(), id, aggregate_type, aggregate_id, event_type,
                           payload, created_at, now(), :error
                    FROM outbox_events WHERE id = :id
                    """
                ),
                {"id": event_id, "error": error},
            )
            await session.execute(
                delete(OutboxEvent).where(OutboxEvent.id == event_id),
            )
        else:
            await session.execute(
                update(OutboxEvent)
                .where(OutboxEvent.id == event_id)
                .values(publish_attempts=attempts, last_error=error),
            )
        await session.commit()
