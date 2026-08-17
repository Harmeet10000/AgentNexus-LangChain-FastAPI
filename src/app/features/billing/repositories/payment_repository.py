"""Payment persistence operations."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from returns.result import Failure, Success
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.shared.result import (
    ConflictAppError,
    InfrastructureAppError,
    NotFoundAppError,
)

from ..models import Payment

if TYPE_CHECKING:
    from decimal import Decimal
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.sql.selectable import Select

    from app.shared.result import (
        AppResult,
    )


class PaymentRepository:
    """Repository for payment transactions."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, payment: Payment) -> AppResult[Payment]:
        try:
            self.session.add(payment)
            await self.session.flush()
            return Success(payment)
        except IntegrityError as exc:
            return Failure(
                ConflictAppError(
                    code="DUPLICATE_PAYMENT",
                    message="Payment already recorded for this Razorpay payment ID",
                    details={"razorpay_payment_id": payment.razorpay_payment_id, "error": str(exc)},
                    source="payment_repository",
                )
            )
        except SQLAlchemyError as exc:
            return Failure(
                InfrastructureAppError(
                    code="DB_ERROR",
                    message="Database error while creating payment",
                    details={"error": str(exc)},
                    source="payment_repository",
                )
            )

    async def find_by_id(self, payment_id: str | UUID) -> AppResult[Payment | None]:
        try:
            statement: Select[tuple[Payment]] = select(Payment).where(Payment.id == payment_id)
            result = await self.session.execute(statement)
            payment = result.scalar_one_or_none()
            if payment is None:
                return Failure(
                    NotFoundAppError(
                        code="PAYMENT_NOT_FOUND",
                        message="Payment not found",
                        details={"payment_id": str(payment_id)},
                        source="payment_repository",
                    )
                )
            return Success(payment)
        except SQLAlchemyError as exc:
            return Failure(
                InfrastructureAppError(
                    code="DB_ERROR",
                    message="Database error while fetching payment",
                    details={"payment_id": str(payment_id), "error": str(exc)},
                    source="payment_repository",
                )
            )

    async def find_by_razorpay_id(self, razorpay_payment_id: str) -> AppResult[Payment | None]:
        try:
            statement: Select[tuple[Payment]] = select(Payment).where(
                Payment.razorpay_payment_id == razorpay_payment_id
            )
            result = await self.session.execute(statement)
            return Success(result.scalar_one_or_none())
        except SQLAlchemyError as exc:
            return Failure(
                InfrastructureAppError(
                    code="DB_ERROR",
                    message="Database error while fetching payment by Razorpay ID",
                    details={"razorpay_payment_id": razorpay_payment_id, "error": str(exc)},
                    source="payment_repository",
                )
            )

    async def find_by_subscription(
        self, subscription_id: str | UUID, *, limit: int = 50, offset: int = 0
    ) -> AppResult[list[Payment]]:
        try:
            statement: Select[tuple[Payment]] = (
                select(Payment)
                .where(Payment.subscription_id == subscription_id)
                .order_by(Payment.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            result = await self.session.execute(statement)
            return Success(list(result.scalars().all()))
        except SQLAlchemyError as exc:
            return Failure(
                InfrastructureAppError(
                    code="DB_ERROR",
                    message="Database error while listing payments for subscription",
                    details={"subscription_id": str(subscription_id), "error": str(exc)},
                    source="payment_repository",
                )
            )

    async def find_by_date_range(
        self, *, date_from: datetime, date_to: datetime
    ) -> AppResult[list[Payment]]:
        try:
            statement: Select[tuple[Payment]] = (
                select(Payment)
                .where(Payment.captured_at >= date_from, Payment.captured_at <= date_to)
                .order_by(Payment.captured_at)
            )
            result = await self.session.execute(statement)
            return Success(list(result.scalars().all()))
        except SQLAlchemyError as exc:
            return Failure(
                InfrastructureAppError(
                    code="DB_ERROR",
                    message="Database error while fetching payments by date range",
                    details={"error": str(exc)},
                    source="payment_repository",
                )
            )

    async def update_refund_amount(
        self, payment: Payment, *, refund_amount: Decimal
    ) -> AppResult[Payment]:
        try:
            statement = (
                update(Payment)
                .where(Payment.id == payment.id)
                .values(refund_amount=refund_amount, updated_at=datetime.now(tz=UTC))
                .returning(Payment)
            )
            result = await self.session.execute(statement)
            updated = result.scalar_one_or_none()
            if updated is None:
                return Failure(
                    NotFoundAppError(
                        code="PAYMENT_NOT_FOUND",
                        message="Payment not found",
                        details={"payment_id": str(payment.id)},
                        source="payment_repository",
                    )
                )
            return Success(updated)
        except SQLAlchemyError as exc:
            return Failure(
                InfrastructureAppError(
                    code="DB_ERROR",
                    message="Database error while updating payment refund amount",
                    details={"payment_id": str(payment.id), "error": str(exc)},
                    source="payment_repository",
                )
            )

    async def update_status(
        self, payment: Payment, *, status: str, extra_values: dict[str, object] | None = None
    ) -> AppResult[Payment]:
        try:  # noqa: PLW0717
            values: dict[str, object] = {"status": status}
            if extra_values:
                values.update(extra_values)
            statement = (
                update(Payment)
                .where(Payment.id == payment.id)
                .values(**values, updated_at=datetime.now(tz=UTC))
                .returning(Payment)
            )
            result = await self.session.execute(statement)
            updated = result.scalar_one_or_none()
            if updated is None:
                return Failure(
                    NotFoundAppError(
                        code="PAYMENT_NOT_FOUND",
                        message="Payment not found",
                        details={"payment_id": str(payment.id)},
                        source="payment_repository",
                    )
                )
            return Success(updated)
        except SQLAlchemyError as exc:
            return Failure(
                InfrastructureAppError(
                    code="DB_ERROR",
                    message="Database error while updating payment status",
                    details={"payment_id": str(payment.id), "error": str(exc)},
                    source="payment_repository",
                )
            )
