"""Transactional outbox relay using PostgreSQL NOTIFY/LISTEN."""

from collections.abc import Awaitable, Callable
from typing import Any

import asyncpg
import asyncpg_listen
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.utils import logger


class OutboxRelay:
    """Listens for outbox events and publishes them to Celery.

    Startup scan (one-shot): catches events created while relay was offline.
    Listen loop (continuous): subscribes to outbox_channel via asyncpg-listen.
    """

    def __init__(
        self,
        database_url: str,
        celery_app: Any,
        *,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._database_url = database_url
        self._celery_app = celery_app
        self._session_factory = session_factory
        self._running = False

    async def run_startup_scan(self) -> None:
        """One-time scan for unpublished events created while relay was offline."""
        engine = create_async_engine(self._database_url)
        try:
            async with engine.begin() as conn:
                rows = (
                    await conn.execute(
                        text(
                            """
                            SELECT id, event_type, payload, publish_attempts
                            FROM outbox_events
                            WHERE published_at IS NULL AND publish_attempts < 5
                            ORDER BY created_at
                            LIMIT 100
                            FOR UPDATE SKIP LOCKED
                            """
                        )
                    )
                ).mappings().all()
        finally:
            await engine.dispose()

        if rows:
            logger.info("outbox_startup_scan", found=len(rows))
            for row in rows:
                await self._publish(dict(row))
        else:
            logger.info("outbox_startup_scan", found=0)

    async def run_listener(self) -> None:
        """Long-running listen loop. Subscribe to outbox_channel, handle notifications."""
        self._running = True
        dsn = self._database_url.replace("+asyncpg", "")
        listener = asyncpg_listen.NotificationListener(
            connect=lambda: asyncpg.connect(dsn=dsn),
        )
        await listener.run(
            handler_per_channel={"outbox_channel": self._handle_notification},
            policy=asyncpg_listen.ListenPolicy.ALL,
            notification_timeout=asyncpg_listen.NO_TIMEOUT,
        )

    async def shutdown(self) -> None:
        """Signal the relay to stop."""
        self._running = False

    async def _handle_notification(
        self,
        notification: asyncpg_listen.Notification | asyncpg_listen.Timeout,
    ) -> None:
        if isinstance(notification, asyncpg_listen.Timeout):
            return
        event_id = notification.payload
        if not event_id:
            return

        async with self._session_factory() as session:
            result = (
                await session.execute(
                    text(
                        """
                        SELECT id, event_type, payload, publish_attempts
                        FROM outbox_events
                        WHERE id = :event_id
                          AND published_at IS NULL
                          AND publish_attempts < 5
                        FOR UPDATE SKIP LOCKED
                        """
                    ),
                    {"event_id": event_id},
                )
            ).mappings().one_or_none()
            if result is None:
                return
            await self._publish(dict(result), session=session)

    async def _publish(
        self,
        row: dict[str, object],
        session: AsyncSession | None = None,
    ) -> None:
        event_id = str(row["id"])
        event_type = str(row["event_type"])
        payload = row["payload"]
        try:
            self._celery_app.send_task(event_type, kwargs=payload)
            await self._mark_published(event_id, session=session)
            logger.info("outbox_published", event_id=event_id, event_type=event_type)
        except Exception as exc:  # noqa: BLE001 — dead-letter on any publish failure
            await self._mark_failed(event_id, str(exc), session=session)
            logger.error("outbox_publish_failed", event_id=event_id, error=str(exc))

    async def _mark_published(
        self,
        event_id: str,
        session: AsyncSession | None = None,
    ) -> None:
        if session is not None:
            await session.execute(
                text("UPDATE outbox_events SET published_at = now() WHERE id = :id"),
                {"id": event_id},
            )
            await session.commit()
        else:
            async with self._session_factory() as s:
                await s.execute(
                    text("UPDATE outbox_events SET published_at = now() WHERE id = :id"),
                    {"id": event_id},
                )
                await s.commit()

    async def _mark_failed(
        self,
        event_id: str,
        error: str,
        session: AsyncSession | None = None,
    ) -> None:
        if session is not None:
            row = (
                await session.execute(
                    text(
                        "SELECT publish_attempts FROM outbox_events WHERE id = :id FOR UPDATE"
                    ),
                    {"id": event_id},
                )
            ).mappings().one_or_none()
            if row is None:
                return
            attempts = row["publish_attempts"] + 1
            if attempts >= 5:
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
                    text("DELETE FROM outbox_events WHERE id = :id"),
                    {"id": event_id},
                )
            else:
                await session.execute(
                    text(
                        "UPDATE outbox_events SET publish_attempts = :attempts, last_error = :error WHERE id = :id"
                    ),
                    {"id": event_id, "attempts": attempts, "error": error},
                )
            await session.commit()
        else:
            async with self._session_factory() as s:
                await self._mark_failed(event_id, error, session=s)
