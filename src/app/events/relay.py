"""Transactional outbox relay using PostgreSQL NOTIFY/LISTEN."""

from __future__ import annotations

import asyncio
from secrets import SystemRandom
from typing import TYPE_CHECKING, cast

import asyncpg
import asyncpg_listen
from asyncpg.exceptions import PostgresError
from celery.exceptions import CeleryError
from pydantic import ValidationError
from sqlalchemy import delete, func, select, text, update

from app.events.model import OutboxEvent
from app.events.writer import with_terminal_outbox
from app.kernel.tenancy import personal_tenant_id
from app.observability.logging import logger
from app.shared.jobs import EnqueueOptions, JobBus, prepare_job, require_direct_receipt
from app.shared.jobs.names import DOCUMENTS_INGEST
from app.utils import ServiceUnavailableException

if TYPE_CHECKING:
    from typing import Final

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from app.shared.jobs import DirectPublicationAdapter

_MAX_RETRIES: Final[int] = 5
_LISTENER_BACKOFF_INITIAL_SECONDS: Final[float] = 0.5
_LISTENER_BACKOFF_MAX_SECONDS: Final[float] = 30.0
_LISTENER_BACKOFF_JITTER_SECONDS: Final[float] = 0.5
_LISTENER_CLOSE_TIMEOUT_SECONDS: Final[float] = 5.0
_JITTER_SOURCE = SystemRandom()


class OutboxRelay:
    """Listens for outbox events and publishes them through the direct seam.

    Startup scan (one-shot): catches events created while relay was offline.
    Listen loop (continuous): subscribes to outbox_channel via asyncpg-listen.
    """

    def __init__(
        self,
        database_url: str,
        publisher: DirectPublicationAdapter | JobBus,
        *,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        """Store the relay's collaborators.

        Args:
            database_url: A plain libpq DSN, as returned by
                `get_database_url(flavour="plain")`. `asyncpg.connect` cannot parse a
                SQLAlchemy dialect scheme, so the URL is used verbatim rather than
                repaired here.
            publisher: Direct publication adapter or JobBus. If a JobBus is
                supplied, only its direct ``enqueue`` path is used; the
                transactional staging path is never involved.
            session_factory: Async session factory for the outbox tables.
        """
        self._database_url = database_url
        self._publisher = publisher
        self._session_factory = session_factory
        self._accepting_notifications = True
        self._in_flight = 0
        self._drained = asyncio.Event()
        self._drained.set()
        self._stop_requested = asyncio.Event()
        self._listener_running = False
        self._listener_connection: asyncpg.Connection | None = None
        self._listener_close_task: asyncio.Task[None] | None = None

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
        if self._listener_running:
            logger.warning("outbox_listener_already_running")
            return

        self._listener_running = True
        failures = 0
        try:
            while not self._stop_requested.is_set():
                if self._listener_connection is not None:
                    failures = await self._retry_listener_connection(failures)
                    continue

                try:
                    listener = asyncpg_listen.NotificationListener(
                        connect=self._connect_listener,
                    )
                    await listener.run(
                        handler_per_channel={"outbox_channel": self._handle_notification},
                        policy=asyncpg_listen.ListenPolicy.ALL,
                        notification_timeout=asyncpg_listen.NO_TIMEOUT,
                    )
                except Exception as exc:  # noqa: BLE001 — listener loop is a survivable degradation boundary
                    failures += 1
                    exc.add_note(f"operation=outbox_listener, failure_count={failures}")
                    logger.bind(
                        error=str(exc),
                        error_type=type(exc).__name__,
                        failure_count=failures,
                    ).exception("outbox_listener_failed")
                else:
                    if self._stop_requested.is_set():
                        break
                    failures += 1
                    logger.bind(failure_count=failures).warning("outbox_listener_stopped")
                finally:
                    await self._close_listener_connection()

                if self._stop_requested.is_set():
                    break

                delay = self._listener_backoff(failures)
                logger.bind(delay_seconds=delay, failure_count=failures).warning(
                    "outbox_listener_reconnecting"
                )
                if await self._wait_for_stop(delay):
                    break
        finally:
            try:
                closed = await self._close_listener_connection()
                if (
                    not closed
                    and self._listener_close_task is None
                    and self._listener_connection is not None
                ):
                    await self._close_listener_connection()
            finally:
                self._listener_running = False

    async def _retry_listener_connection(self, failures: int) -> int:
        if await self._close_listener_connection():
            return failures
        failures += 1
        delay = self._listener_backoff(failures)
        logger.bind(delay_seconds=delay, failure_count=failures).warning(
            "outbox_listener_connection_retry"
        )
        await self._wait_for_stop(delay)
        return failures

    async def _connect_listener(self) -> asyncpg.Connection:
        """Connect while retaining ownership for cleanup if the listener fails."""
        connection = await asyncpg.connect(dsn=self._database_url)
        self._listener_connection = connection
        return connection

    async def _close_listener_connection(self) -> bool:
        connection = self._listener_connection
        if connection is None:
            return True

        close_task = self._listener_close_task
        try:
            is_closed = getattr(connection, "is_closed", None)
            if callable(is_closed) and is_closed():
                self._listener_connection = None
                return True
            if close_task is None:
                close_task = asyncio.create_task(connection.close())
                self._listener_close_task = close_task
                close_task.add_done_callback(
                    lambda task: self._listener_close_finished(task, connection)
                )
            await asyncio.wait_for(
                asyncio.shield(close_task), timeout=_LISTENER_CLOSE_TIMEOUT_SECONDS
            )
        except asyncio.CancelledError:
            if close_task is not None and close_task.cancelled():
                self._listener_close_task = None
                return False
            raise
        except Exception as exc:  # noqa: BLE001 — cleanup must not stop listener re-entry
            exc.add_note("operation=close_outbox_listener_connection")
            logger.bind(error=str(exc), error_type=type(exc).__name__).warning(
                "outbox_listener_connection_close_failed"
            )
            if close_task is not None and close_task.done():
                self._listener_close_task = None
            return False
        else:
            if self._listener_close_task is close_task:
                self._listener_close_task = None
            if self._listener_connection is connection:
                self._listener_connection = None
            return True

    def _listener_close_finished(
        self, task: asyncio.Task[None], connection: asyncpg.Connection
    ) -> None:
        if self._listener_close_task is task:
            self._listener_close_task = None
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            error.add_note("operation=close_outbox_listener_connection")
            logger.bind(error=str(error), error_type=type(error).__name__).warning(
                "outbox_listener_connection_close_failed"
            )
            return
        if self._listener_connection is connection:
            self._listener_connection = None

    @staticmethod
    def _listener_backoff(failure_count: int) -> float:
        exponential = _LISTENER_BACKOFF_INITIAL_SECONDS * 2 ** (failure_count - 1)
        bounded = min(exponential, _LISTENER_BACKOFF_MAX_SECONDS)
        jitter = _JITTER_SOURCE.uniform(0.0, min(_LISTENER_BACKOFF_JITTER_SECONDS, bounded))
        return min(bounded + jitter, _LISTENER_BACKOFF_MAX_SECONDS)

    async def _wait_for_stop(self, timeout_seconds: float) -> bool:
        if self._stop_requested.is_set():
            return True
        try:
            await asyncio.wait_for(self._stop_requested.wait(), timeout=timeout_seconds)
        except TimeoutError:
            return False
        return True

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
        self._stop_requested.set()
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
            job = prepare_job(event_type, payload)
            try:
                if isinstance(self._publisher, JobBus):
                    receipt = await self._publisher.enqueue(job, EnqueueOptions())
                else:
                    receipt = await self._publisher.publish(job, options=EnqueueOptions())
            except ValidationError as exc:
                exc.add_note(f"event_id={event_id}, event_type={event_type}")
                message = "Direct publication returned no valid task id"
                raise ServiceUnavailableException(message) from exc
            require_direct_receipt(receipt, operation="outbox_relay_publish")
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
                    select(
                        OutboxEvent.publish_attempts,
                        OutboxEvent.event_type,
                        OutboxEvent.aggregate_id,
                        OutboxEvent.payload,
                    )
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
            if row["event_type"] == DOCUMENTS_INGEST:
                # The task may be dead-lettered before its worker starts.
                # Emit the same terminal integration event in this transaction
                # so a document cannot remain stuck in ``received`` forever.
                source_payload = cast("dict[str, object]", row["payload"])
                raw_version = source_payload.get("document_version", 1)
                document_version = raw_version if isinstance(raw_version, int) else 1
                raw_user_id = str(source_payload.get("user_id") or "")
                raw_tenant_id = str(source_payload.get("tenant_id") or "")
                resolved_tenant_id = raw_tenant_id or (
                    personal_tenant_id(raw_user_id) if raw_user_id else ""
                )
                await with_terminal_outbox(
                    session,
                    document_id=str(source_payload.get("document_id") or row["aggregate_id"]),
                    payload={
                        "document_id": str(
                            source_payload.get("document_id") or row["aggregate_id"]
                        ),
                        "user_id": raw_user_id,
                        "tenant_id": resolved_tenant_id,
                        "document_version": document_version,
                        "status": "failed",
                        "attempt": attempts,
                        # The original outbox event is the stable task
                        # identity for a broker dead-letter.
                        "task_id": event_id,
                        "error_code": "OUTBOX_DEAD_LETTERED",
                        "error_message": error,
                    },
                )
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
