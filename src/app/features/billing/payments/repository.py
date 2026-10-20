"""Payment persistence operations."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from returns.result import Failure, Success
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.kernel.result import add_database_error_note
from app.observability.logging import trace_layer

from .exceptions import PaymentConflictError, PaymentInfrastructureError, PaymentNotFoundError
from .models import Payment

if TYPE_CHECKING:
    from decimal import Decimal
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.sql.selectable import Select

    from .exceptions import PaymentResult


class PaymentRepository:
    """Repository for payment transactions."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    @trace_layer("repository")
    async def create(self, payment: Payment) -> PaymentResult[Payment]:
        try:
            self.session.add(payment)
            await self.session.flush()
            return Success(payment)
        except IntegrityError as exc:
            add_database_error_note(exc, table="payments")
            await self.session.rollback()
            return Failure(
                PaymentConflictError(
                    message="Payment already recorded for this Razorpay payment ID",
                    details={"razorpay_payment_id": payment.razorpay_payment_id, "error": str(exc)},
                    source="payment_repository",
                )
            )
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="payments")
            await self.session.rollback()
            return Failure(
                PaymentInfrastructureError(
                    message="Database error while creating payment",
                    details={"error": str(exc)},
                    source="payment_repository",
                )
            )

    @trace_layer("repository")
    async def find_by_id(self, payment_id: str | UUID) -> PaymentResult[Payment | None]:
        try:
            statement: Select[tuple[Payment]] = select(Payment).where(Payment.id == payment_id)
            result = await self.session.execute(statement)
            payment = result.scalar_one_or_none()
            if payment is None:
                return Failure(
                    PaymentNotFoundError(
                        message="Payment not found",
                        details={"payment_id": str(payment_id)},
                        source="payment_repository",
                    )
                )
            return Success(payment)
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="payments")
            await self.session.rollback()
            return Failure(
                PaymentInfrastructureError(
                    message="Database error while fetching payment",
                    details={"payment_id": str(payment_id), "error": str(exc)},
                    source="payment_repository",
                )
            )

    @trace_layer("repository")
    async def find_by_id_for_update(self, payment_id: str | UUID) -> PaymentResult[Payment | None]:
        """Load a payment holding a row lock for a validate-and-update sequence.

        The lock is held until the surrounding transaction commits or rolls
        back, so two concurrent refunds cannot both pass the remaining-amount
        check against the same snapshot.
        """
        try:
            statement: Select[tuple[Payment]] = (
                select(Payment).where(Payment.id == payment_id).with_for_update()
            )
            result = await self.session.execute(statement)
            payment = result.scalar_one_or_none()
            if payment is None:
                return Failure(
                    PaymentNotFoundError(
                        message="Payment not found",
                        details={"payment_id": str(payment_id)},
                        source="payment_repository",
                    )
                )
            return Success(payment)
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="payments")
            await self.session.rollback()
            return Failure(
                PaymentInfrastructureError(
                    message="Database error while locking payment",
                    details={"payment_id": str(payment_id), "error": str(exc)},
                    source="payment_repository",
                )
            )

    @trace_layer("repository")
    async def find_by_razorpay_id(self, razorpay_payment_id: str) -> PaymentResult[Payment | None]:
        try:
            statement: Select[tuple[Payment]] = select(Payment).where(
                Payment.razorpay_payment_id == razorpay_payment_id
            )
            result = await self.session.execute(statement)
            return Success(result.scalar_one_or_none())
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="payments")
            await self.session.rollback()
            return Failure(
                PaymentInfrastructureError(
                    message="Database error while fetching payment by Razorpay ID",
                    details={"razorpay_payment_id": razorpay_payment_id, "error": str(exc)},
                    source="payment_repository",
                )
            )

    @trace_layer("repository")
    async def find_by_razorpay_id_for_update(
        self, razorpay_payment_id: str
    ) -> PaymentResult[Payment | None]:
        """Row-locked variant for webhook redelivery races on the same payment."""
        try:
            statement: Select[tuple[Payment]] = (
                select(Payment)
                .where(Payment.razorpay_payment_id == razorpay_payment_id)
                .with_for_update()
            )
            result = await self.session.execute(statement)
            return Success(result.scalar_one_or_none())
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="payments")
            await self.session.rollback()
            return Failure(
                PaymentInfrastructureError(
                    message="Database error while locking payment by Razorpay ID",
                    details={"razorpay_payment_id": razorpay_payment_id, "error": str(exc)},
                    source="payment_repository",
                )
            )

    @trace_layer("repository")
    async def find_by_subscription(
        self, subscription_id: str | UUID, *, limit: int = 50, offset: int = 0
    ) -> PaymentResult[list[Payment]]:
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
            add_database_error_note(exc, table="payments")
            await self.session.rollback()
            return Failure(
                PaymentInfrastructureError(
                    message="Database error while listing payments for subscription",
                    details={"subscription_id": str(subscription_id), "error": str(exc)},
                    source="payment_repository",
                )
            )

    @trace_layer("repository")
    async def find_refundable_captured_by_subscription(
        self, subscription_id: str | UUID
    ) -> PaymentResult[list[Payment]]:
        """Load every captured payment with remaining refundable value.

        Cancellation is an economic operation, not a paginated listing.  A
        failed/newest provider attempt must never hide a captured payment, so
        this query deliberately has no LIMIT/OFFSET and orders the current
        captured payment first.  The final id tie-breaker keeps selection
        deterministic when timestamps collide.
        """
        try:
            statement: Select[tuple[Payment]] = (
                select(Payment)
                .where(
                    Payment.subscription_id == subscription_id,
                    Payment.status.in_(("captured", "partially_refunded")),
                    Payment.amount > (Payment.refund_amount * 100),
                )
                .order_by(
                    Payment.captured_at.desc().nullslast(),
                    Payment.created_at.desc(),
                    Payment.id.desc(),
                )
            )
            result = await self.session.execute(statement)
            return Success(list(result.scalars().all()))
        except SQLAlchemyError as exc:
            add_database_error_note(
                exc,
                table="payments",
                operation="find_refundable_captured_by_subscription",
            )
            await self.session.rollback()
            return Failure(
                PaymentInfrastructureError(
                    message="Database error while finding refundable captured payments",
                    details={"subscription_id": str(subscription_id), "error": str(exc)},
                    source="payment_repository",
                )
            )

    @trace_layer("repository")
    async def find_oldest_captured_invoice_by_subscription(
        self, subscription_id: str | UUID
    ) -> PaymentResult[UUID | None]:
        """Return the oldest captured payment's inverse invoice binding.

        The payment ``invoice_id`` is the authoritative inverse relation.  A
        direct ordered query avoids both fabricated invoice ids and the
        100-row pagination trap in downgrade source selection.
        """
        try:
            statement = (
                select(Payment.invoice_id)
                .where(
                    Payment.subscription_id == subscription_id,
                    Payment.status.in_(("captured", "partially_refunded", "refunded")),
                    Payment.invoice_id.is_not(None),
                )
                # PostgreSQL sorts NULLs first for ASC.  An old captured row
                # without a capture timestamp must not outrank a dated
                # capture when selecting the source invoice for a downgrade.
                .order_by(
                    Payment.captured_at.asc().nullslast(),
                    Payment.created_at.asc(),
                    Payment.id.asc(),
                )
                .limit(1)
            )
            result = await self.session.execute(statement)
            return Success(result.scalar_one_or_none())
        except SQLAlchemyError as exc:
            add_database_error_note(
                exc,
                table="payments",
                operation="find_oldest_captured_invoice_by_subscription",
            )
            await self.session.rollback()
            return Failure(
                PaymentInfrastructureError(
                    message="Database error while resolving the source invoice",
                    details={"subscription_id": str(subscription_id), "error": str(exc)},
                    source="payment_repository",
                )
            )

    @trace_layer("repository")
    async def find_by_date_range(
        self, *, date_from: datetime, date_to: datetime
    ) -> PaymentResult[list[Payment]]:
        try:
            statement: Select[tuple[Payment]] = (
                select(Payment)
                .where(Payment.captured_at >= date_from, Payment.captured_at <= date_to)
                .order_by(Payment.captured_at)
            )
            result = await self.session.execute(statement)
            return Success(list(result.scalars().all()))
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="payments")
            await self.session.rollback()
            return Failure(
                PaymentInfrastructureError(
                    message="Database error while fetching payments by date range",
                    details={"error": str(exc)},
                    source="payment_repository",
                )
            )

    @trace_layer("repository")
    async def update_refund_amount(
        self, payment: Payment, *, refund_amount: Decimal
    ) -> PaymentResult[Payment]:
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
                    PaymentNotFoundError(
                        message="Payment not found",
                        details={"payment_id": str(payment.id)},
                        source="payment_repository",
                    )
                )
            return Success(updated)
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="payments")
            await self.session.rollback()
            return Failure(
                PaymentInfrastructureError(
                    message="Database error while updating payment refund amount",
                    details={"payment_id": str(payment.id), "error": str(exc)},
                    source="payment_repository",
                )
            )

    @trace_layer("repository")
    async def update_status(
        self, payment: Payment, *, status: str, extra_values: dict[str, object] | None = None
    ) -> PaymentResult[Payment]:
        try:
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
                    PaymentNotFoundError(
                        message="Payment not found",
                        details={"payment_id": str(payment.id)},
                        source="payment_repository",
                    )
                )
            return Success(updated)
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="payments")
            await self.session.rollback()
            return Failure(
                PaymentInfrastructureError(
                    message="Database error while updating payment status",
                    details={"payment_id": str(payment.id), "error": str(exc)},
                    source="payment_repository",
                )
            )

    @trace_layer("repository")
    async def link_invoice(self, payment: Payment, invoice_id: UUID) -> PaymentResult[Payment]:
        """Set the payment's invoice inverse relation exactly once.

        Invoice generation and this update run in the same SQLAlchemy session;
        callers therefore get an all-or-nothing invoice/document transaction.
        The predicate also makes a concurrent repair harmless when it is
        attempting to write the same invoice, while refusing to overwrite an
        already authoritative, different relation.
        """
        try:
            statement = (
                update(Payment)
                .where(
                    Payment.id == payment.id,
                    (Payment.invoice_id.is_(None)) | (Payment.invoice_id == invoice_id),
                )
                .values(invoice_id=invoice_id, updated_at=datetime.now(tz=UTC))
                .returning(Payment)
            )
            result = await self.session.execute(statement)
            linked = result.scalar_one_or_none()
            if linked is not None:
                return Success(linked)

            current = await self.session.execute(
                select(Payment.invoice_id).where(Payment.id == payment.id)
            )
            current_invoice_id = current.scalar_one_or_none()
            if current_invoice_id == invoice_id:
                payment.invoice_id = invoice_id
                return Success(payment)
            if current_invoice_id is None:
                return Failure(
                    PaymentNotFoundError(
                        message="Payment not found while linking invoice",
                        details={"payment_id": str(payment.id)},
                        source="payment_repository",
                    )
                )
            return Failure(
                PaymentConflictError(
                    message="Payment is already linked to a different invoice",
                    details={
                        "payment_id": str(payment.id),
                        "invoice_id": str(invoice_id),
                        "existing_invoice_id": str(current_invoice_id),
                    },
                    source="payment_repository",
                )
            )
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="payments", operation="link_invoice")
            await self.session.rollback()
            return Failure(
                PaymentInfrastructureError(
                    message="Database error while linking payment to invoice",
                    details={"payment_id": str(payment.id), "invoice_id": str(invoice_id)},
                    source="payment_repository",
                )
            )
