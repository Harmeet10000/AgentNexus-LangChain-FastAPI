"""Invoice generation, voiding, and listing (GST-compliant)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from returns.result import Failure

from app.config import get_settings
from app.features.billing.models import (
    AuditAction,
    AuditLog,
    Invoice,
    InvoiceLineItem,
    InvoiceStatus,
    InvoiceVoid,
    PaymentReceipt,
)
from app.features.billing.response import failure_envelope
from app.features.billing.services.pdf import render_invoice_pdf, render_receipt_pdf
from app.features.billing.tax import (
    paisa_to_rupees,
    split_tax_for_gst,
    split_tax_inclusive,
    state_code_from_gstin,
)
from app.utils import ValidationException

from ..dto import InvoiceLineItemDTO, InvoiceResponse

if TYPE_CHECKING:
    from uuid import UUID

    from app.features.billing.models import Payment, Plan, Subscription
    from app.features.billing.repositories import BillingRepositories
    from app.features.billing.response import ServiceResult
    from app.shared.services.storage import StorageService

    from ..dto import VoidInvoiceDTO

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


class InvoiceService:
    """Create and manage GST-compliant invoices."""

    def __init__(
        self,
        repos: BillingRepositories,
        storage: StorageService | None = None,
    ) -> None:
        self.repos = repos
        self.storage = storage

    async def generate_for_payment(  # noqa: PLR0914
        self,
        payment: Payment,
        subscription: Subscription,
        plan: Plan,
        *,
        buyer_gstin: str | None = None,
    ) -> Invoice:
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

        number_result = await self.repos.invoices.generate_invoice_number(
            prefix=settings.BILLING_INVOICE_PREFIX, year=datetime.now(tz=UTC).year
        )
        if isinstance(number_result, Failure):
            raise ValidationException(number_result.failure().message)
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
        create_result = await self.repos.invoices.create(invoice)
        if isinstance(create_result, Failure):
            raise ValidationException(create_result.failure().message)
        created = create_result.unwrap()

        pdf_url = await self._store_pdf(render_invoice_pdf(created), created.invoice_number)
        if pdf_url:
            update_result = await self.repos.invoices.update_status(
                created, status=created.status, extra_values={"pdf_url": pdf_url}
            )
            if not isinstance(update_result, Failure):
                created.pdf_url = pdf_url

        await self.repos.audit.create(
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
        return created

    async def generate_receipt_for_payment(
        self,
        payment: Payment,
        subscription: Subscription,
        plan: Plan,
    ) -> PaymentReceipt:
        """Create a non-taxable payment receipt (Requirement 36)."""
        settings = get_settings()
        number_result = await self.repos.invoices.generate_receipt_number(
            prefix=settings.BILLING_RECEIPT_PREFIX, year=datetime.now(tz=UTC).year
        )
        if isinstance(number_result, Failure):
            raise ValidationException(number_result.failure().message)

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
        self.repos.session.add(receipt)
        await self.repos.session.flush()

        pdf_url = await self._store_pdf(
            render_receipt_pdf(receipt), receipt.receipt_number, folder="billing/receipts"
        )
        if pdf_url:
            receipt.pdf_url = pdf_url
            await self.repos.session.flush()
        return receipt

    async def list_invoices(
        self,
        user_id: str,
        *,
        subscription_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> ServiceResult[list[InvoiceResponse]]:
        result = await self.repos.invoices.list_by_user(
            user_id,
            subscription_id=subscription_id,
            status=status,
            limit=limit,
            offset=offset,
        )
        if isinstance(result, Failure):
            return failure_envelope(result.failure(), operation="list_invoices")
        return [_invoice_to_response(i) for i in result.unwrap()]

    async def get_invoice(
        self, invoice_id: str | UUID, *, user_id: str | None = None
    ) -> ServiceResult[InvoiceResponse]:
        result = await self.repos.invoices.find_by_id(invoice_id)
        if isinstance(result, Failure):
            return failure_envelope(result.failure(), operation="get_invoice")
        invoice = result.unwrap()
        if invoice is None:
            msg = "Invoice not found"
            raise ValidationException(msg)
        if user_id is not None and invoice.user_id != user_id:
            msg = "Invoice does not belong to this user"
            raise ValidationException(msg)
        return _invoice_to_response(invoice)

    async def void_invoice(  # noqa: PLR0912
        self, invoice_id: str | UUID, dto: VoidInvoiceDTO, *, user_id: str
    ) -> ServiceResult[InvoiceResponse]:
        """Void an issued/paid invoice and optionally reissue (Requirement 41)."""
        result = await self.repos.invoices.find_by_id(invoice_id)
        if isinstance(result, Failure):
            return failure_envelope(result.failure(), operation="void_invoice")
        invoice = result.unwrap()
        if invoice is None:
            msg = "Invoice not found"
            raise ValidationException(msg)
        if invoice.user_id != user_id:
            msg = "Invoice does not belong to this user"
            raise ValidationException(msg)
        if invoice.status == InvoiceStatus.VOID.value:
            msg = "Invoice is already void"
            raise ValidationException(msg)
        if invoice.status not in {InvoiceStatus.ISSUED.value, InvoiceStatus.PAID.value}:
            msg = f"Cannot void invoice in status '{invoice.status}'"
            raise ValidationException(msg)

        now = datetime.now(tz=UTC)
        self.repos.session.add(
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
        void_result = await self.repos.invoices.update_status(
            invoice, status=InvoiceStatus.VOID.value
        )
        if isinstance(void_result, Failure):
            return failure_envelope(void_result.failure(), operation="void_invoice")
        voided = void_result.unwrap()

        await self.repos.audit.create(
            AuditLog(
                entity_type="invoice",
                entity_id=str(voided.id),
                action=AuditAction.INVOICE_VOIDED.value,
                user_id=user_id,
                changes={"reason": dto.reason, "description": dto.description},
            )
        )

        if dto.reissue:
            sub_result = await self.repos.subscriptions.find_by_id(voided.subscription_id)
            if isinstance(sub_result, Failure):
                return failure_envelope(sub_result.failure(), operation="void_invoice")
            subscription = sub_result.unwrap()
            if subscription is None:
                msg = "Subscription not found"
                raise ValidationException(msg)
            plan_result = await self.repos.plans.find_by_id(subscription.plan_id)
            if isinstance(plan_result, Failure):
                return failure_envelope(plan_result.failure(), operation="void_invoice")
            plan = plan_result.unwrap()
            if plan is None:
                msg = "Plan not found"
                raise ValidationException(msg)
            payment = None
            if voided.payment_id is not None:
                payment_result = await self.repos.payments.find_by_id(voided.payment_id)
                if isinstance(payment_result, Failure):
                    return failure_envelope(payment_result.failure(), operation="void_invoice")
                payment = payment_result.unwrap()
            if payment is None:
                msg = "Cannot reissue without an associated payment"
                raise ValidationException(msg)

            reissued = await self.generate_for_payment(payment, subscription, plan)
            await self.repos.audit.create(
                AuditLog(
                    entity_type="invoice",
                    entity_id=str(reissued.id),
                    action=AuditAction.INVOICE_REISSUED.value,
                    user_id=user_id,
                    changes={"original_invoice_id": str(voided.id)},
                )
            )
            return _invoice_to_response(reissued)
        return _invoice_to_response(voided)

    async def _store_pdf(
        self, pdf_bytes: bytes, name: str, *, folder: str = "billing/invoices"
    ) -> str | None:
        if self.storage is None:
            return None
        key = f"{folder}/{name}.pdf"
        await self.storage.put_object(
            key=key, data=pdf_bytes, content_type="application/pdf", metadata={}
        )
        public_base = get_settings().S3_PUBLIC_URL.rstrip("/")
        return f"{public_base}/{key}" if public_base else None
