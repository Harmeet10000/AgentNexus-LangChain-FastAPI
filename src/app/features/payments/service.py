"""Payment record and refund service."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from returns.result import Failure

from app.features.audit.model import AuditAction, AuditLog
from app.features.invoices.tax import paisa_to_rupees
from app.shared.result import app_error_to_exception, log_expected_failure
from app.utils import NotFoundException, ValidationException

from .clients.razorpay_client import RazorpayClient
from .dto import PaymentResponse, RefundResponse
from .model import Payment, PaymentStatus

if TYPE_CHECKING:
    from uuid import UUID

    from app.features.audit.repository import AuditLogRepository
    from app.features.payments.repository import PaymentRepository
    from app.features.subscriptions.model import Subscription
    from app.shared.result import AppError

    from .dto import PaymentRecordDTO, RefundRequestDTO


def _payment_to_response(payment: Payment) -> PaymentResponse:
    return PaymentResponse(
        id=str(payment.id),
        subscription_id=str(payment.subscription_id),
        invoice_id=str(payment.invoice_id) if payment.invoice_id else None,
        razorpay_payment_id=payment.razorpay_payment_id,
        amount=payment.amount,
        currency=payment.currency,
        status=payment.status,
        method=payment.method,
        captured_at=payment.captured_at,
        failed_at=payment.failed_at,
        error_code=payment.error_code,
        error_description=payment.error_description,
        refund_amount=payment.refund_amount,
        created_at=payment.created_at,
    )


def _repo_failure(error: AppError, operation: str) -> None:
    log_expected_failure(error, operation=operation)
    raise app_error_to_exception(error)


class PaymentService:
    """Record payments, list them per subscription, and issue refunds."""

    def __init__(
        self,
        payments: PaymentRepository,
        audit: AuditLogRepository,
        razorpay: RazorpayClient | None = None,
    ) -> None:
        self.payments = payments
        self.audit = audit
        self.razorpay = razorpay or RazorpayClient()

    async def record_payment(self, dto: PaymentRecordDTO, *, subscription: Subscription) -> Payment:
        """Persist a captured (or failed) payment idempotently.

        Used by the webhook service. Raises project exceptions on failure
        (this is orchestration code, not a router-facing service method).
        """
        existing = await self.payments.find_by_razorpay_id(dto.razorpay_payment_id)
        if isinstance(existing, Failure):
            raise ValidationException(existing.failure().message)
        existing_payment = existing.unwrap()
        if existing_payment is not None:
            return existing_payment

        captured = dto.captured_at or datetime.now(tz=UTC)
        payment = Payment(
            subscription_id=subscription.id,
            razorpay_payment_id=dto.razorpay_payment_id,
            razorpay_order_id=dto.razorpay_order_id,
            amount=dto.amount,
            currency=dto.currency,
            status=PaymentStatus.CAPTURED.value,
            method=dto.method.value if dto.method else None,
            captured_at=captured,
            metadata_=dto.metadata,
        )
        result = await self.payments.create(payment)
        if isinstance(result, Failure):
            raise ValidationException(result.failure().message)
        return result.unwrap()

    async def record_failed_payment(
        self,
        *,
        razorpay_payment_id: str,
        subscription_id: str,
        error_code: str,
        error_description: str,
    ) -> Payment:
        payment = Payment(
            subscription_id=subscription_id,
            razorpay_payment_id=razorpay_payment_id,
            amount=0,
            currency="INR",
            status=PaymentStatus.FAILED.value,
            failed_at=datetime.now(tz=UTC),
            error_code=error_code,
            error_description=error_description,
        )
        result = await self.payments.create(payment)
        if isinstance(result, Failure):
            raise ValidationException(result.failure().message)
        return result.unwrap()

    async def list_payments(
        self,
        subscription_id: str | UUID,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[PaymentResponse]:
        result = await self.payments.find_by_subscription(
            subscription_id, limit=limit, offset=offset
        )
        if isinstance(result, Failure):
            _repo_failure(result.failure(), "list_payments")
        return [_payment_to_response(p) for p in result.unwrap()]

    async def get_payment(self, payment_id: str) -> PaymentResponse:
        result = await self.payments.find_by_id(payment_id)
        if isinstance(result, Failure):
            _repo_failure(result.failure(), "get_payment")
        payment = result.unwrap()
        if payment is None:
            msg = "Payment"
            raise NotFoundException(msg, payment_id)
        return _payment_to_response(payment)

    async def refund(
        self, payment_id: str, dto: RefundRequestDTO, *, user_id: str
    ) -> RefundResponse:
        result = await self.payments.find_by_id(payment_id)
        if isinstance(result, Failure):
            _repo_failure(result.failure(), "refund")
        payment = result.unwrap()
        if payment is None:
            msg = "Payment"
            raise NotFoundException(msg, payment_id)
        if payment.status not in {
            PaymentStatus.CAPTURED.value,
            PaymentStatus.PARTIALLY_REFUNDED.value,
        }:
            msg = f"Cannot refund a payment in status '{payment.status}'"
            raise ValidationException(msg)
        if dto.amount > payment.amount - self._refund_paisa(payment):
            msg = "Refund amount exceeds the unrefunded payment amount"
            raise ValidationException(msg)

        razorpay_refund = await self.razorpay.create_refund(
            payment_id=payment.razorpay_payment_id,
            amount=dto.amount,
            notes={"reason": dto.reason or ""} if dto.reason else None,
        )
        refund_id: str = razorpay_refund.get("id", "")
        if not refund_id:
            msg = "Razorpay did not return a refund id"
            raise ValidationException(msg)

        new_refund_paisa = self._refund_paisa(payment) + dto.amount
        new_status = (
            PaymentStatus.REFUNDED.value
            if new_refund_paisa >= payment.amount
            else PaymentStatus.PARTIALLY_REFUNDED.value
        )
        result = await self.payments.update_status(
            payment,
            status=new_status,
            extra_values={"refund_amount": paisa_to_rupees(new_refund_paisa)},
        )
        if isinstance(result, Failure):
            _repo_failure(result.failure(), "refund")
        updated = result.unwrap()

        await self.audit.create(
            AuditLog(
                entity_type="payment",
                entity_id=str(updated.id),
                action=AuditAction.REFUND_ISSUED.value,
                user_id=user_id,
                changes={"razorpay_refund_id": refund_id, "amount": dto.amount},
            )
        )
        return RefundResponse(
            id=str(updated.id),
            razorpay_refund_id=refund_id,
            payment_id=str(updated.id),
            amount=dto.amount,
            currency=updated.currency,
            status=new_status,
            created_at=datetime.now(tz=UTC),
        )

    @staticmethod
    def _refund_paisa(payment: Payment) -> int:
        return int((payment.refund_amount or Decimal(0)) * 100)

    async def handle_refund_processed(
        self, *, razorpay_payment_id: str, refund_paisa: int
    ) -> None:
        """Finalize a payment after ``refund.processed`` (Requirement 11)."""
        result = await self.payments.find_by_razorpay_id(razorpay_payment_id)
        if isinstance(result, Failure):
            return
        payment = result.unwrap()
        if payment is None:
            return
        new_refund_paisa = self._refund_paisa(payment) + refund_paisa
        new_status = (
            PaymentStatus.REFUNDED.value
            if new_refund_paisa >= payment.amount
            else PaymentStatus.PARTIALLY_REFUNDED.value
        )
        update = await self.payments.update_status(
            payment,
            status=new_status,
            extra_values={"refund_amount": paisa_to_rupees(new_refund_paisa)},
        )
        if isinstance(update, Failure):
            return
        updated = update.unwrap()
        await self.audit.create(
            AuditLog(
                entity_type="payment",
                entity_id=str(updated.id),
                action=AuditAction.REFUND_PROCESSED.value,
                changes={"refund_paisa": refund_paisa, "status": new_status},
            )
        )

    async def handle_chargeback(
        self, *, razorpay_payment_id: str, dispute_id: str, reason: str
    ) -> None:
        """Record a chargeback on ``payment.dispute.created`` (Requirement 11).

        There is no dedicated payment status for disputes; the payment stays
        CAPTURED and the dispute is recorded in ``metadata_`` + audit trail.
        """
        result = await self.payments.find_by_razorpay_id(razorpay_payment_id)
        if isinstance(result, Failure):
            return
        payment = result.unwrap()
        if payment is None:
            return
        update = await self.payments.update_status(
            payment,
            status=payment.status,
            extra_values={
                "metadata_": {
                    **(payment.metadata_ or {}),
                    "dispute": {"dispute_id": dispute_id, "reason": reason},
                }
            },
        )
        if isinstance(update, Failure):
            return
        updated = update.unwrap()
        await self.audit.create(
            AuditLog(
                entity_type="payment",
                entity_id=str(updated.id),
                action=AuditAction.CHARGEBACK.value,
                changes={"dispute_id": dispute_id, "reason": reason},
            )
        )
