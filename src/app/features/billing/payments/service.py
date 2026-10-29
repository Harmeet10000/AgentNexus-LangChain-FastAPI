"""Payment record and refund service."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Protocol, cast

from returns.result import Failure, Success

from app.features.audit.models import AuditAction, AuditLog
from app.features.billing.money import paisa_to_rupees
from app.features.billing.providers import CreateRefundOperation, ProviderReceipt
from app.observability.logging import logger, trace_layer

from .clients.razorpay_client import (
    CircuitOpenError,
    RazorpayPermanentError,
    RazorpayRetryableError,
)
from .dto import PaymentResponse, RefundRequestDTO, RefundResponse
from .exceptions import (
    PaymentCollaboratorError,
    PaymentManualReviewError,
    PaymentNotFoundError,
    PaymentProviderError,
    PaymentProviderUnavailableError,
    PaymentValidationError,
)
from .models import Payment, PaymentStatus

if TYPE_CHECKING:
    from uuid import UUID

    from app.features.audit.repository import AuditLogRepository
    from app.features.billing.payments.repository import PaymentRepository
    from app.features.billing.subscriptions.coordinator import SubscriptionLifecycleCoordinator
    from app.features.billing.subscriptions.models import Subscription

    from .dto import PaymentRecordDTO
    from .exceptions import PaymentResult


class _LegacyRefundProvider(Protocol):
    async def create_refund(
        self, *, payment_id: str, amount: int, notes: dict[str, object]
    ) -> dict[str, object]: ...


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


_PROCESSED_REFUND_IDS_KEY = "processed_refund_ids"


def _recorded_refund_ids(payment: Payment) -> list[str]:
    metadata = payment.metadata_ or {}
    recorded = metadata.get(_PROCESSED_REFUND_IDS_KEY, [])
    if not isinstance(recorded, list):
        return []
    return [entry for entry in recorded if isinstance(entry, str)]


def _with_recorded_refund(payment: Payment, refund_id: str | None) -> dict[str, object]:
    """Merge one refund id into the payment's idempotency ledger.

    Stored in ``metadata_`` so no schema change is needed. Callers hold a
    row lock (``find_by_*_for_update``), so the read-merge-write here is
    serialized per payment.
    """
    metadata = dict(payment.metadata_ or {})
    recorded = _recorded_refund_ids(payment)
    if refund_id and refund_id not in recorded:
        recorded.append(refund_id)
    metadata[_PROCESSED_REFUND_IDS_KEY] = recorded
    return metadata


class PaymentService:
    """Record payments, list them per subscription, and issue refunds."""

    def __init__(
        self,
        payments: PaymentRepository,
        audit: AuditLogRepository,
        *,
        lifecycle_coordinator: SubscriptionLifecycleCoordinator | None = None,
        razorpay: object | None = None,
    ) -> None:
        self.payments = payments
        self.audit = audit
        # ``razorpay`` is retained only for isolated legacy unit doubles.  The
        # production dependency graph never supplies it; all real mutations
        # must pass through the lifespan provider/coordinator seam below.
        self._legacy_razorpay = razorpay
        self.lifecycle_coordinator = lifecycle_coordinator

    @trace_layer("service")
    async def record_payment(
        self, dto: PaymentRecordDTO, *, subscription: Subscription
    ) -> PaymentResult[Payment]:
        """Persist a captured (or failed) payment idempotently.

        Used by the webhook service. Raises project exceptions on failure
        (this is orchestration code, not a router-facing service method).
        """
        existing = await self.payments.find_by_razorpay_id(dto.razorpay_payment_id)
        if isinstance(existing, Failure):
            return existing
        existing_payment = existing.unwrap()
        if existing_payment is not None:
            return Success(existing_payment)

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
            return result
        return Success(result.unwrap())

    @trace_layer("service")
    async def record_failed_payment(
        self,
        *,
        razorpay_payment_id: str,
        subscription_id: str,
        error_code: str,
        error_description: str,
    ) -> PaymentResult[Payment]:
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
            return result
        return Success(result.unwrap())

    @trace_layer("service")
    async def list_payments(
        self,
        subscription_id: str | UUID,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> PaymentResult[list[PaymentResponse]]:
        result = await self.payments.find_by_subscription(
            subscription_id, limit=limit, offset=offset
        )
        if isinstance(result, Failure):
            return result
        return Success([_payment_to_response(p) for p in result.unwrap()])

    @trace_layer("service")
    async def get_payment(self, payment_id: str) -> PaymentResult[PaymentResponse]:
        result = await self.payments.find_by_id(payment_id)
        if isinstance(result, Failure):
            return result
        payment = result.unwrap()
        if payment is None:
            return Failure(
                PaymentNotFoundError(
                    message="Payment not found", details={"payment_id": payment_id}
                )
            )
        return Success(_payment_to_response(payment))

    @trace_layer("service")
    async def refund(  # noqa: PLR0914 -- provider/idempotency stages are explicit
        self,
        payment_id: str,
        dto: RefundRequestDTO,
        *,
        user_id: str,
        idempotency_key: str | None = None,
    ) -> PaymentResult[RefundResponse]:
        result = await self.payments.find_by_id_for_update(payment_id)
        if isinstance(result, Failure):
            return result
        payment = result.unwrap()
        if payment is None:
            return Failure(
                PaymentNotFoundError(
                    message="Payment not found", details={"payment_id": payment_id}
                )
            )
        if payment.status not in {
            PaymentStatus.CAPTURED.value,
            PaymentStatus.PARTIALLY_REFUNDED.value,
        }:
            return Failure(
                PaymentValidationError(
                    message=f"Cannot refund a payment in status '{payment.status}'"
                )
            )
        if dto.amount > payment.amount - self._refund_paisa(payment):
            return Failure(
                PaymentValidationError(
                    message="Refund amount exceeds the unrefunded payment amount"
                )
            )

        # A keyless request is one explicit, stable operation for this payment.
        # Partial refunds that are intentionally separate must provide distinct
        # Idempotency-Key values; refunded_amount is mutable and cannot identify
        # an operation across a retry.
        operation_key = idempotency_key or f"billing:refund:{payment.id}:default"
        correlation_id = f"{operation_key}:provider"
        operation_notes = {key: str(value) for key, value in (dto.notes or {}).items()}
        if dto.reason:
            operation_notes["reason"] = dto.reason
        operation = CreateRefundOperation(
            payment_id=payment.razorpay_payment_id,
            amount=dto.amount,
            correlation_id=correlation_id,
            notes=operation_notes,
        )
        provider_result = await self._execute_refund_provider(
            operation, subscription_id=str(payment.subscription_id), idempotency_key=operation_key
        )
        if isinstance(provider_result, Failure):
            return provider_result
        value = getattr(provider_result.unwrap(), "value", None)
        if not isinstance(value, ProviderReceipt):
            return Failure(
                PaymentProviderError(
                    message="Razorpay did not return a refund id", source="razorpay"
                )
            )
        provider_currency = getattr(value, "currency", None)
        currency_mismatch = provider_currency is not None and provider_currency != payment.currency
        if (
            value.payment_id != payment.razorpay_payment_id
            or value.amount != dto.amount
            or currency_mismatch
        ):
            # The provider mutation has already produced a durable receipt.  Do
            # not project it locally when its identity/economic facts disagree
            # with the locked payment; the coordinator's operation evidence and
            # the receipt details below are the reconciliation record.
            details: dict[str, object] = {
                "manual_review": True,
                "requires_reconciliation": True,
                "operation": "create_refund",
                "idempotency_key": operation_key,
                "correlation_id": correlation_id,
                "provider_refund_id": value.provider_refund_id,
                "expected_payment_id": payment.razorpay_payment_id,
                "provider_payment_id": value.payment_id,
                "expected_amount_paisa": dto.amount,
                "provider_amount_paisa": value.amount,
            }
            if provider_currency is not None:
                details["expected_currency"] = payment.currency
                details["provider_currency"] = provider_currency
            logger.bind(
                operation="refund",
                payment_id=str(payment.id),
                provider_refund_id=value.provider_refund_id,
                correlation_id=correlation_id,
            ).warning("Provider refund receipt requires reconciliation")
            return Failure(
                PaymentManualReviewError(
                    message="Provider refund receipt does not match the requested payment",
                    details=details,
                    source="payment_service",
                )
            )
        refund_id = value.provider_refund_id

        if refund_id in _recorded_refund_ids(payment):
            logger.bind(operation="refund", refund_id=refund_id).info(
                "Refund receipt already applied; skipping local balance update"
            )
            current_refund_paisa = self._refund_paisa(payment)
            current_status = (
                PaymentStatus.REFUNDED.value
                if current_refund_paisa >= payment.amount
                else PaymentStatus.PARTIALLY_REFUNDED.value
            )
            return Success(
                RefundResponse(
                    id=refund_id,
                    razorpay_refund_id=refund_id,
                    payment_id=str(payment.id),
                    amount=dto.amount,
                    currency=payment.currency,
                    status=current_status,
                    created_at=datetime.now(tz=UTC),
                )
            )

        new_refund_paisa = self._refund_paisa(payment) + dto.amount
        new_status = (
            PaymentStatus.REFUNDED.value
            if new_refund_paisa >= payment.amount
            else PaymentStatus.PARTIALLY_REFUNDED.value
        )
        result = await self.payments.update_status(
            payment,
            status=new_status,
            extra_values={
                "refund_amount": paisa_to_rupees(new_refund_paisa),
                "metadata_": _with_recorded_refund(payment, refund_id),
            },
        )
        if isinstance(result, Failure):
            return result
        updated = result.unwrap()

        audit_result = await self.audit.create(
            AuditLog(
                entity_type="payment",
                entity_id=str(updated.id),
                action=AuditAction.REFUND_ISSUED.value,
                user_id=user_id,
                changes={"razorpay_refund_id": refund_id, "amount": dto.amount},
            )
        )
        if isinstance(audit_result, Failure):
            error = audit_result.failure()
            return Failure(PaymentCollaboratorError(message=error.message, details=error.details))
        return Success(
            RefundResponse(
                id=refund_id,
                razorpay_refund_id=refund_id,
                payment_id=str(updated.id),
                amount=dto.amount,
                currency=updated.currency,
                status=new_status,
                created_at=datetime.now(tz=UTC),
            )
        )

    @trace_layer("service")
    async def settle_credit_application(
        self,
        payment: Payment,
        *,
        invoice_id: str,
        credit_application_key: str,
        amount_paisa: int,
        user_id: str,
    ) -> PaymentResult[RefundResponse]:
        """Refund the cash equivalent of one durable invoice credit application.

        The operation key is derived solely from the immutable invoice/payment
        application identity.  It therefore survives webhook replay and lets
        the provider coordinator correlate an unknown outcome without issuing
        a second refund.
        """
        if amount_paisa <= 0:
            return Failure(
                PaymentValidationError(
                    message="Credit settlement amount must be positive",
                    details={"amount_paisa": amount_paisa},
                )
            )
        operation_key = (
            f"billing:credit-settlement:{invoice_id}:{payment.id}:{credit_application_key}"
        )
        return await self.refund(
            str(payment.id),
            RefundRequestDTO(
                amount=amount_paisa,
                reason="invoice_credit_application",
                notes={"invoice_id": invoice_id, "credit_application_key": credit_application_key},
            ),
            user_id=user_id,
            idempotency_key=operation_key,
        )

    async def _execute_refund_provider(
        self,
        operation: CreateRefundOperation,
        *,
        subscription_id: str,
        idempotency_key: str,
    ) -> PaymentResult[object]:
        if self.lifecycle_coordinator is not None:
            result = await self.lifecycle_coordinator.execute_provider_operation(
                operation,
                subscription_id=subscription_id,
                idempotency_key=idempotency_key,
            )
            if isinstance(result, Failure):
                error = result.failure()
                provider_code = getattr(error, "provider_code", None)
                code = getattr(provider_code, "value", None)
                if code is not None and code in {"unavailable", "rate_limited"}:
                    return Failure(
                        PaymentProviderUnavailableError(message=error.message, source="razorpay")
                    )
                return Failure(PaymentProviderError(message=error.message, source="razorpay"))
            return Success(result.unwrap())
        if self._legacy_razorpay is None:
            return Failure(
                PaymentProviderError(
                    message="Billing provider coordinator is not configured", source="billing"
                )
            )
        try:
            payload = await cast("_LegacyRefundProvider", self._legacy_razorpay).create_refund(
                payment_id=operation.payment_id,
                amount=operation.amount,
                notes=dict(operation.notes),
            )
        except (RazorpayRetryableError, CircuitOpenError) as exc:
            return Failure(PaymentProviderUnavailableError(message=str(exc), source="razorpay"))
        except RazorpayPermanentError as exc:
            return Failure(PaymentProviderError(message=str(exc), source="razorpay"))
        refund_id = payload.get("id", "")
        if not isinstance(refund_id, str) or not refund_id:
            return Failure(
                PaymentProviderError(
                    message="Razorpay did not return a refund id", source="razorpay"
                )
            )
        return Success(
            type(
                "_LegacyProviderResult",
                (),
                {
                    "value": ProviderReceipt(
                        provider_refund_id=refund_id,
                        payment_id=operation.payment_id,
                        amount=operation.amount,
                    )
                },
            )()
        )

    @staticmethod
    def _refund_paisa(payment: Payment) -> int:
        return int((payment.refund_amount or Decimal(0)) * 100)

    @trace_layer("service")
    async def handle_refund_processed(
        self, *, razorpay_payment_id: str, refund_paisa: int, refund_id: str | None = None
    ) -> PaymentResult[None]:
        """Finalize a payment after ``refund.processed`` (Requirement 11).

        Idempotent on the Razorpay refund id: the same delivery (or a retry)
        records each refund once, so ``refund()`` creating the refund and the
        webhook confirming it no longer double-count.
        """
        result = await self.payments.find_by_razorpay_id_for_update(razorpay_payment_id)
        if isinstance(result, Failure):
            return result
        payment = result.unwrap()
        if payment is None:
            return Success(None)
        if refund_id is None:
            logger.bind(
                operation="handle_refund_processed", payment_id=razorpay_payment_id
            ).warning("Refund webhook has no provider refund id; manual review required")
            return Failure(
                PaymentManualReviewError(
                    message="Refund webhook is missing its provider refund id",
                    details={"payment_id": razorpay_payment_id, "manual_review": True},
                )
            )
        if refund_id in _recorded_refund_ids(payment):
            logger.bind(operation="handle_refund_processed", refund_id=refund_id).info(
                "Refund already recorded, skipping duplicate delivery"
            )
            return Success(None)
        remaining_refundable_paisa = payment.amount - self._refund_paisa(payment)
        if refund_paisa <= 0 or refund_paisa > remaining_refundable_paisa:
            logger.bind(
                operation="handle_refund_processed",
                payment_id=razorpay_payment_id,
                refund_id=refund_id,
                refund_paisa=refund_paisa,
                remaining_refundable_paisa=remaining_refundable_paisa,
            ).warning("Refund webhook amount requires reconciliation")
            return Failure(
                PaymentManualReviewError(
                    message="Refund webhook amount is outside the refundable payment balance",
                    details={
                        "manual_review": True,
                        "requires_reconciliation": True,
                        "payment_id": razorpay_payment_id,
                        "refund_id": refund_id,
                        "refund_paisa": refund_paisa,
                        "remaining_refundable_paisa": remaining_refundable_paisa,
                    },
                    source="payment_service",
                )
            )
        new_refund_paisa = self._refund_paisa(payment) + refund_paisa
        new_status = (
            PaymentStatus.REFUNDED.value
            if new_refund_paisa >= payment.amount
            else PaymentStatus.PARTIALLY_REFUNDED.value
        )
        update = await self.payments.update_status(
            payment,
            status=new_status,
            extra_values={
                "refund_amount": paisa_to_rupees(new_refund_paisa),
                "metadata_": _with_recorded_refund(payment, refund_id),
            },
        )
        if isinstance(update, Failure):
            return update
        updated = update.unwrap()
        audit_result = await self.audit.create(
            AuditLog(
                entity_type="payment",
                entity_id=str(updated.id),
                action=AuditAction.REFUND_PROCESSED.value,
                changes={"refund_paisa": refund_paisa, "status": new_status},
            )
        )
        if isinstance(audit_result, Failure):
            error = audit_result.failure()
            return Failure(PaymentCollaboratorError(message=error.message, details=error.details))
        return Success(None)

    @trace_layer("service")
    async def handle_chargeback(
        self, *, razorpay_payment_id: str, dispute_id: str, reason: str
    ) -> PaymentResult[None]:
        """Record a chargeback on ``payment.dispute.created`` (Requirement 11).

        There is no dedicated payment status for disputes; the payment stays
        CAPTURED and the dispute is recorded in ``metadata_`` + audit trail.
        """
        result = await self.payments.find_by_razorpay_id(razorpay_payment_id)
        if isinstance(result, Failure):
            return result
        payment = result.unwrap()
        if payment is None:
            return Success(None)
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
            return update
        updated = update.unwrap()
        audit_result = await self.audit.create(
            AuditLog(
                entity_type="payment",
                entity_id=str(updated.id),
                action=AuditAction.CHARGEBACK.value,
                changes={"dispute_id": dispute_id, "reason": reason},
            )
        )
        if isinstance(audit_result, Failure):
            error = audit_result.failure()
            return Failure(PaymentCollaboratorError(message=error.message, details=error.details))
        return Success(None)
