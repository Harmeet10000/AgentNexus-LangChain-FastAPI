"""Webhook event persistence for idempotency and replay."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from returns.result import Failure, Success
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from .errors import WebhookConflictError, WebhookInfrastructureError, WebhookNotFoundError
from .model import WebhookEvent

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.sql.selectable import Select

    from .errors import WebhookResult


class WebhookEventRepository:
    """Repository for webhook event logging and idempotency."""

    def __init__(self, session: AsyncSession) -> None:
        self.session: AsyncSession = session

    async def create(self, event: WebhookEvent) -> WebhookResult[WebhookEvent]:
        try:
            self.session.add(event)
            await self.session.flush()
            return Success(event)
        except IntegrityError as exc:
            await self.session.rollback()
            return Failure(
                WebhookConflictError(
                    message="Webhook event already processed",
                    details={"razorpay_event_id": event.razorpay_event_id, "error": str(exc)},
                    source="webhook_event_repository",
                )
            )
        except SQLAlchemyError as exc:
            await self.session.rollback()
            return Failure(
                WebhookInfrastructureError(
                    message="Database error while creating webhook event",
                    details={"error": str(exc)},
                    source="webhook_event_repository",
                )
            )

    async def find_by_razorpay_event_id(
        self, razorpay_event_id: str
    ) -> WebhookResult[WebhookEvent | None]:
        try:
            statement: Select[tuple[WebhookEvent]] = select(WebhookEvent).where(
                WebhookEvent.razorpay_event_id == razorpay_event_id
            )
            result = await self.session.execute(statement)
            return Success(result.scalar_one_or_none())
        except SQLAlchemyError as exc:
            await self.session.rollback()
            return Failure(
                WebhookInfrastructureError(
                    message="Database error while checking webhook event",
                    details={"razorpay_event_id": razorpay_event_id, "error": str(exc)},
                    source="webhook_event_repository",
                )
            )

    async def find_by_id(self, event_id: str | UUID) -> WebhookResult[WebhookEvent | None]:
        try:
            statement: Select[tuple[WebhookEvent]] = select(WebhookEvent).where(
                WebhookEvent.id == event_id
            )
            result = await self.session.execute(statement)
            event = result.scalar_one_or_none()
            if event is None:
                return Failure(
                    WebhookNotFoundError(
                        message="Webhook event not found",
                        details={"event_id": str(event_id)},
                        source="webhook_event_repository",
                    )
                )
            return Success(event)
        except SQLAlchemyError as exc:
            await self.session.rollback()
            return Failure(
                WebhookInfrastructureError(
                    message="Database error while fetching webhook event",
                    details={"event_id": str(event_id), "error": str(exc)},
                    source="webhook_event_repository",
                )
            )

    async def find_failed_events(self, *, limit: int = 100) -> WebhookResult[list[WebhookEvent]]:
        try:
            statement: Select[tuple[WebhookEvent]] = (
                select(WebhookEvent)
                .where(WebhookEvent.status == "failed")
                .order_by(WebhookEvent.failed_at.desc())
                .limit(limit)
            )
            result = await self.session.execute(statement)
            return Success(list(result.scalars().all()))
        except SQLAlchemyError as exc:
            await self.session.rollback()
            return Failure(
                WebhookInfrastructureError(
                    message="Database error while listing failed webhook events",
                    details={"error": str(exc)},
                    source="webhook_event_repository",
                )
            )

    async def update_status(
        self,
        event: WebhookEvent,
        *,
        status: str,
        extra_values: dict[str, object] | None = None,
    ) -> WebhookResult[WebhookEvent]:
        try:
            values: dict[str, object] = {"status": status}
            if extra_values:
                values.update(extra_values)
            statement = (
                update(WebhookEvent)
                .where(WebhookEvent.id == event.id)
                .values(**values, updated_at=datetime.now(tz=UTC))
                .returning(WebhookEvent)
            )
            result = await self.session.execute(statement)
            updated = result.scalar_one_or_none()
            if updated is None:
                return Failure(
                    WebhookNotFoundError(
                        message="Webhook event not found",
                        details={"event_id": str(event.id)},
                        source="webhook_event_repository",
                    )
                )
            return Success(updated)
        except SQLAlchemyError as exc:
            await self.session.rollback()
            return Failure(
                WebhookInfrastructureError(
                    message="Database error while updating webhook event status",
                    details={"event_id": str(event.id), "error": str(exc)},
                    source="webhook_event_repository",
                )
            )
