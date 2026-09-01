"""Invoice generation, voiding, and listing (GST-compliant)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from returns.result import Failure, Success
from sqlalchemy.exc import SQLAlchemyError

from app.config import get_settings
from app.features.audit.model import AuditAction, AuditLog
from app.shared.result.errors import ErrorKind

from .dto import InvoiceLineItemDTO, InvoiceResponse
from .errors import (
    InvoiceCollaboratorError,
    InvoiceConflictError,
    InvoiceInfrastructureError,
    InvoiceNotFoundError,
    InvoiceStorageError,
    InvoiceValidationError,
)
from .invoice_void import InvoiceVoid
from .model import (
    Invoice,
    InvoiceLineItem,
    InvoiceStatus,
)
from .pdf import render_invoice_pdf, render_receipt_pdf
from .receipt import PaymentReceipt
from .tax import (
    paisa_to_rupees,
    split_tax_for_gst,
    split_tax_inclusive,
    state_code_from_gstin,
)

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from app.features.audit.repository import AuditLogRepository
    from app.features.invoices.repository import InvoiceRepository
    from app.features.payments.model import Payment
    from app.features.payments.repository import PaymentRepository
    from app.features.plans.model import Plan
    from app.features.plans.repository import PlanRepository
    from app.features.subscriptions.model import Subscription
    from app.features.subscriptions.repository import SubscriptionRepository
    from app.shared.result.errors import FeatureError
    from app.shared.services.storage import StorageService

    from .dto import VoidInvoiceDTO
    from .errors import InvoiceError, InvoiceResult

_DEFAULT_SAC_CODE = "998314"


def _invoice_to_response(invoice: Invoice) -> InvoiceResponse:
    return InvoiceResponse(
        id=str(invoice.id),
        invoice_number=invoice.invoice_number,
        subscription_id=str(invoice.subscription_id),
        payment_id=str(invoice.payment_id) if invoice.payment_id else None,
        status=invoice.status,
        subtotal=invoice.subtotal,
        tax_rate=invoice.tax_rate,
        tax_amount=invoice.tax_amount,
        total=invoice.total,
        currency=invoice.currency,
        seller_gstin=invoice.seller_gstin,
        buyer_gstin=invoice.buyer_gstin,
        place_of_supply=invoice.place_of_supply,
        sac_code=invoice.sac_code,
        cgst_amount=invoice.cgst_amount,
        sgst_amount=invoice.sgst_amount,
        igst_amount=invoice.igst_amount,
        issued_at=invoice.issued_at,
        due_at=invoice.due_at,
        paid_at=invoice.paid_at,
        pdf_url=invoice.pdf_url,
        line_items=[
            InvoiceLineItemDTO(
                plan_name=item.plan_name,
                description=item.description,
                quantity=item.quantity,
                unit_price=item.unit_price,
                amount=item.amount,
                tax_amount=item.tax_amount,
                sac_code=item.sac_code,
            )
            for item in invoice.line_items
        ],
        created_at=invoice.created_at,
    )


def _translate_collaborator_error(error: FeatureError) -> InvoiceError:
    if error.kind == ErrorKind.NOT_FOUND:
        translated = InvoiceNotFoundError(message=error.message, details=error.details)
    elif error.kind == ErrorKind.CONFLICT:
        translated = InvoiceConflictError(message=error.message, details=error.details)
    elif error.kind == ErrorKind.VALIDATION:
        translated = InvoiceValidationError(message=error.message, details=error.details)
    else:
        translated = InvoiceCollaboratorError(message=error.message, details=error.details)
    return translated


class InvoiceService:
    """Create and manage GST-compliant invoices."""

    def __init__(  # noqa: PLR0917
        self,
        session: AsyncSession,
        invoices: InvoiceRepository,
        subscriptions: SubscriptionRepository,
        plans: PlanRepository,
        payments: PaymentRepository,
        audit: AuditLogRepository,
        storage: StorageService | None = None,
    ) -> None:
        self.session = session
        self.invoices = invoices
        self.subscriptions = subscriptions
        self.plans = plans
        self.payments = payments
        self.audit = audit
        self.storage = storage

    async def generate_for_payment(  # noqa: PLR0914
        self,
        payment: Payment,
        subscription: Subscription,
        plan: Plan,
        *,
        buyer_gstin: str | None = None,
    ) -> InvoiceResult[Invoice]:
        """Create a GST invoice for a captured payment.

        GST-inclusive split (Property 1, Requirements 12/38): all amounts are
        computed in integer paisa so ``invoice.total * 100 == payment.amount``
        exactly, then converted to rupees for the stored columns.
        """
        settings = get_settings()
        subtotal_paisa, tax_paisa = split_tax_inclusive(payment.amount, plan.tax_rate)
        seller_state = state_code_from_gstin(settings.BILLING_SELLER_GSTIN)
        buyer_state = state_code_from_gstin(buyer_gstin) if buyer_gstin else None
        cgst_paisa, sgst_paisa, igst_paisa = split_tax_for_gst(
            tax_paisa,
            seller_state_code=seller_state or "99",
            buyer_state_code=buyer_state,
        )

        number_result = await self.invoices.generate_invoice_number(
            prefix=settings.BILLING_INVOICE_PREFIX, year=datetime.now(tz=UTC).year
        )
        if isinstance(number_result, Failure):
            return number_result
        invoice_number = number_result.unwrap()

        now = datetime.now(tz=UTC)
        invoice = Invoice(
            invoice_number=invoice_number,
            subscription_id=subscription.id,
            payment_id=payment.id,
            user_id=subscription.user_id,
            status=InvoiceStatus.PAID.value,
            subtotal=paisa_to_rupees(subtotal_paisa),
            tax_rate=plan.tax_rate,
            tax_amount=paisa_to_rupees(tax_paisa),
            total=paisa_to_rupees(payment.amount),
            currency=payment.currency,
            seller_gstin=settings.BILLING_SELLER_GSTIN,
            buyer_gstin=buyer_gstin,
            place_of_supply=buyer_state or settings.BILLING_PLACE_OF_SUPPLY,
            sac_code=_DEFAULT_SAC_CODE,
            cgst_amount=paisa_to_rupees(cgst_paisa),
            sgst_amount=paisa_to_rupees(sgst_paisa),
            igst_amount=paisa_to_rupees(igst_paisa),
            issued_at=now,
            due_at=now + timedelta(days=7),
            paid_at=payment.captured_at,
            line_items=[
                InvoiceLineItem(
                    plan_name=plan.name,
                    description=plan.description,
                    quantity=1,
                    unit_price=paisa_to_rupees(subtotal_paisa),
                    amount=paisa_to_rupees(subtotal_paisa),
                    tax_amount=paisa_to_rupees(tax_paisa),
                    sac_code=_DEFAULT_SAC_CODE,
                )
            ],
            metadata_={
                "razorpay_payment_id": payment.razorpay_payment_id,
                "razorpay_subscription_id": subscription.razorpay_subscription_id,
            },
        )
        create_result = await self.invoices.create(invoice)
        if isinstance(create_result, Failure):
            return create_result
        created = create_result.unwrap()

        pdf_result = await self._store_pdf(render_invoice_pdf(created), created.invoice_number)
        if isinstance(pdf_result, Failure):
            return pdf_result
        pdf_url = pdf_result.unwrap()
        if pdf_url:
            update_result = await self.invoices.update_status(
                created, status=created.status, extra_values={"pdf_url": pdf_url}
            )
            if isinstance(update_result, Failure):
                return update_result
            created.pdf_url = pdf_url

        audit_result = await self.audit.create(
            AuditLog(
                entity_type="invoice",
                entity_id=str(created.id),
                action=AuditAction.INVOICE_GENERATED.value,
                user_id=subscription.user_id,
                changes={
                    "invoice_number": created.invoice_number,
                    "total": str(created.total),
                    "payment_id": str(payment.id),
                },
            )
        )
        if isinstance(audit_result, Failure):
            # ponytail: audit shares the same session/transaction as the invoice.
            # If audit flush fails, repository rollback has already undone the invoice
            # (sourcery broader_impact, ADR D8). Swallowing would commit a clean tx
            # with no invoice persisted, so we must surface the failure.
            return Failure(_translate_collaborator_error(audit_result.failure()))
        return Success(created)

    async def generate_receipt_for_payment(
        self,
        payment: Payment,
        subscription: Subscription,
        plan: Plan,
    ) -> InvoiceResult[PaymentReceipt]:
        """Create a non-taxable payment receipt (Requirement 36)."""
        settings = get_settings()
        number_result = await self.invoices.generate_receipt_number(
            prefix=settings.BILLING_RECEIPT_PREFIX, year=datetime.now(tz=UTC).year
        )
        if isinstance(number_result, Failure):
            return number_result

        receipt = PaymentReceipt(
            receipt_number=number_result.unwrap(),
            subscription_id=subscription.id,
            payment_id=payment.id,
            user_id=subscription.user_id,
            amount=paisa_to_rupees(payment.amount),
            currency=payment.currency,
            payment_method=payment.method,
            razorpay_payment_id=payment.razorpay_payment_id,
            receipt_date=payment.captured_at or datetime.now(tz=UTC),
            billing_period_start=subscription.current_period_start,
            billing_period_end=subscription.current_period_end,
            plan_name=plan.name,
        )
        try:
            self.session.add(receipt)
            await self.session.flush()
        except SQLAlchemyError as exc:
            await self.session.rollback()
            return Failure(
                InvoiceInfrastructureError(
                    message="Database error while creating payment receipt",
                    details={"error": str(exc)},
                )
            )

        pdf_result = await self._store_pdf(
            render_receipt_pdf(receipt), receipt.receipt_number, folder="billing/receipts"
        )
        if isinstance(pdf_result, Failure):
            return pdf_result
        pdf_url = pdf_result.unwrap()
        if pdf_url:
            receipt.pdf_url = pdf_url
            await self.session.flush()
        return Success(receipt)

    async def list_invoices(
        self,
        user_id: str,
        *,
        subscription_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> InvoiceResult[list[InvoiceResponse]]:
        result = await self.invoices.list_by_user(
            user_id,
            subscription_id=subscription_id,
            status=status,
            limit=limit,
            offset=offset,
        )
        if isinstance(result, Failure):
            return result
        return Success([_invoice_to_response(i) for i in result.unwrap()])

    async def get_invoice(
        self, invoice_id: str | UUID, *, user_id: str | None = None
    ) -> InvoiceResult[InvoiceResponse]:
        result = await self.invoices.find_by_id(invoice_id)
        if isinstance(result, Failure):
            return result
        invoice = result.unwrap()
        if invoice is None:
            return Failure(InvoiceNotFoundError(message="Invoice not found"))
        if user_id is not None and invoice.user_id != user_id:
            return Failure(InvoiceValidationError(message="Invoice does not belong to this user"))
        return Success(_invoice_to_response(invoice))

    async def void_invoice(  # noqa: PLR0912
        self, invoice_id: str | UUID, dto: VoidInvoiceDTO, *, user_id: str
    ) -> InvoiceResult[InvoiceResponse]:
        """Void an issued/paid invoice and optionally reissue (Requirement 41)."""
        result = await self.invoices.find_by_id(invoice_id)
        if isinstance(result, Failure):
            return result
        invoice = result.unwrap()
        if invoice is None:
            return Failure(InvoiceNotFoundError(message="Invoice not found"))
        if invoice.user_id != user_id:
            return Failure(InvoiceValidationError(message="Invoice does not belong to this user"))
        if invoice.status == InvoiceStatus.VOID.value:
            return Failure(InvoiceValidationError(message="Invoice is already void"))
        if invoice.status not in {InvoiceStatus.ISSUED.value, InvoiceStatus.PAID.value}:
            return Failure(
                InvoiceValidationError(message=f"Cannot void invoice in status '{invoice.status}'")
            )

        now = datetime.now(tz=UTC)
        self.session.add(
            InvoiceVoid(
                original_invoice_id=invoice.id,
                void_reason=dto.reason,
                void_description=dto.description,
                voided_by_user_id=user_id,
                voided_at=now,
                original_invoice_number=invoice.invoice_number,
                original_subtotal=invoice.subtotal,
                original_tax_rate=invoice.tax_rate,
                original_tax_amount=invoice.tax_amount,
                original_total=invoice.total,
                original_currency=invoice.currency,
            )
        )
        void_result = await self.invoices.update_status(invoice, status=InvoiceStatus.VOID.value)
        if isinstance(void_result, Failure):
            return void_result
        voided: Invoice = void_result.unwrap()

        audit_void = await self.audit.create(
            AuditLog(
                entity_type="invoice",
                entity_id=str(voided.id),
                action=AuditAction.INVOICE_VOIDED.value,
                user_id=user_id,
                changes={"reason": dto.reason, "description": dto.description},
            )
        )
        if isinstance(audit_void, Failure):
            return Failure(_translate_collaborator_error(audit_void.failure()))

        if dto.reissue:
            sub_result = await self.subscriptions.find_by_id(voided.subscription_id)
            if isinstance(sub_result, Failure):
                return Failure(_translate_collaborator_error(sub_result.failure()))
            subscription = sub_result.unwrap()
            if subscription is None:
                return Failure(InvoiceNotFoundError(message="Subscription not found"))
            plan_result = await self.plans.find_by_id(subscription.plan_id)
            if isinstance(plan_result, Failure):
                return Failure(_translate_collaborator_error(plan_result.failure()))
            plan = plan_result.unwrap()
            if plan is None:
                return Failure(InvoiceNotFoundError(message="Plan not found"))
            payment = None
            if voided.payment_id is not None:
                payment_result = await self.payments.find_by_id(voided.payment_id)
                if isinstance(payment_result, Failure):
                    return Failure(_translate_collaborator_error(payment_result.failure()))
                payment = payment_result.unwrap()
            if payment is None:
                return Failure(
                    InvoiceValidationError(message="Cannot reissue without an associated payment")
                )

            reissued_result = await self.generate_for_payment(payment, subscription, plan)
            if isinstance(reissued_result, Failure):
                return reissued_result
            reissued = reissued_result.unwrap()
            audit_reissue = await self.audit.create(
                AuditLog(
                    entity_type="invoice",
                    entity_id=str(reissued.id),
                    action=AuditAction.INVOICE_REISSUED.value,
                    user_id=user_id,
                    changes={"original_invoice_id": str(voided.id)},
                )
            )
            if isinstance(audit_reissue, Failure):
                return Failure(_translate_collaborator_error(audit_reissue.failure()))
            return Success(_invoice_to_response(reissued))
        return Success(_invoice_to_response(voided))

    async def _store_pdf(
        self, pdf_bytes: bytes, name: str, *, folder: str = "billing/invoices"
    ) -> InvoiceResult[str | None]:
        if self.storage is None:
            return Success(None)
        key = f"{folder}/{name}.pdf"
        storage_result = await self.storage.put_object(
            key=key, data=pdf_bytes, content_type="application/pdf", metadata={}
        )
        if isinstance(storage_result, Failure):
            error = storage_result.failure()
            return Failure(
                InvoiceStorageError(
                    message=error.message,
                    details=error.details,
                    source="storage",
                )
            )
        public_base = get_settings().S3_PUBLIC_URL.rstrip("/")
        return Success(f"{public_base}/{key}" if public_base else None)
