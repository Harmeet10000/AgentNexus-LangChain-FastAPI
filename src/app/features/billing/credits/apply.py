"""Atomic application of the credit ledger to an invoice."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field
from returns.result import Failure, Success

from app.features.audit.models import AuditLog
from app.features.billing.credits.exceptions import (
    CreditAuthorizationError,
    CreditCollaboratorError,
    CreditConflictError,
    CreditManualReviewError,
    CreditNotFoundError,
    CreditValidationError,
)
from app.features.billing.credits.models import (
    CreditConsumption,
    CreditNoteApplication,
    CreditStatus,
)
from app.features.billing.invoices.models import InvoiceStatus
from app.features.billing.money import paisa_to_rupees, rupees_to_paisa
from app.features.billing.providers import ProviderReceipt  # noqa: TC001

if TYPE_CHECKING:
    from collections.abc import Callable
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from app.features.audit.repository import AuditLogRepository
    from app.features.billing.credits.exceptions import CreditResult
    from app.features.billing.credits.repository import ConsumptionRepository, CreditRepository
    from app.features.billing.invoices.repository import InvoiceRepository


class _CreditApplyAbortError(Exception):
    """Internal control-flow exception which forces an owner transaction to roll back.

    ``returns.Failure`` is deliberately not used as a transaction signal:
    returning it from ``async with session.begin()`` tells SQLAlchemy that the
    block succeeded, so a partially-applied credit ledger would be committed.
    The exception never crosses the public method boundary.
    """

    def __init__(self, error: object) -> None:
        super().__init__(str(error))
        self.error = error


class ConsumedCredit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    credit_id: str = Field(serialization_alias="creditId")
    consumed_amount: int = Field(serialization_alias="consumedAmount")
    remaining_balance: int = Field(serialization_alias="remainingBalance")


class CreditApplication(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    credit_applied_paisa: int = Field(serialization_alias="creditAppliedPaisa")
    credit_applied_rupees: float = Field(serialization_alias="creditAppliedRupees")
    cash_due_paisa: int = Field(serialization_alias="cashDuePaisa")
    cash_due_rupees: float = Field(serialization_alias="cashDueRupees")
    credits_consumed: list[ConsumedCredit] = Field(serialization_alias="creditsConsumed")
    invoice_paid_in_full: bool = Field(serialization_alias="invoicePaidInFull")


class CreditApplicationItem(BaseModel):
    """One exact ledger row selected by a settlement preview.

    The balances are part of the token on purpose.  A preview is a snapshot,
    not a reservation; the apply step must prove that every row still has the
    same balance before it consumes anything.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    credit_id: str = Field(serialization_alias="creditId")
    credit_note_id: str | None = Field(default=None, serialization_alias="creditNoteId")
    amount_paisa: int = Field(serialization_alias="amountPaisa")
    expected_credit_balance: int = Field(serialization_alias="expectedCreditBalance")
    expected_note_balance: int | None = Field(
        default=None, serialization_alias="expectedNoteBalance"
    )


class CreditApplicationPlan(BaseModel):
    """Optimistic, replayable plan prepared before a provider refund.

    It contains no mutable ORM objects and is safe to carry across the
    provider call.  ``apply_settled`` re-locks and compares these facts before
    invoking the existing atomic ledger path.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    invoice_id: str = Field(serialization_alias="invoiceId")
    user_id: str = Field(serialization_alias="userId")
    gross_paisa: int = Field(serialization_alias="grossPaisa")
    credit_applied_paisa: int = Field(serialization_alias="creditAppliedPaisa")
    items: list[CreditApplicationItem]

    @property
    def settlement_key(self) -> str:
        return f"invoice:{self.invoice_id}"


class ApplyCreditsToInvoice:
    """Apply authoritative notes and standalone grants in one transaction."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        credit_repo_factory: Callable[[AsyncSession], CreditRepository],
        consumptions_factory: Callable[[AsyncSession], ConsumptionRepository],
        invoices_factory: Callable[[AsyncSession], InvoiceRepository],
        audit_factory: Callable[[AsyncSession], AuditLogRepository],
    ) -> None:
        self._session_factory = session_factory
        self._credit_repo_factory = credit_repo_factory
        self._consumptions_factory = consumptions_factory
        self._invoices_factory = invoices_factory
        self._audit_factory = audit_factory

    async def prepare(self, invoice_id: UUID, user_id: str) -> CreditResult[CreditApplicationPlan]:
        """Read and lock a deterministic credit snapshot without consuming it.

        Webhook settlement uses this seam before asking the provider for a
        refund.  The transaction is intentionally read-only from the domain's
        perspective; its locks are released before the remote call and the
        returned plan is revalidated with CAS during :meth:`apply_settled`.
        """
        session = self._session_factory()
        try:
            async with session, session.begin():
                return await self._prepare_with_session(session, invoice_id, user_id)
        except _CreditApplyAbortError as abort:
            return Failure(abort.error)  # type: ignore[arg-type]

    async def preview(self, invoice_id: UUID, user_id: str) -> CreditResult[CreditApplicationPlan]:
        """Named alias for callers that treat plans as read-only previews."""
        return await self.prepare(invoice_id, user_id)

    async def reserve(self, invoice_id: UUID, user_id: str) -> CreditResult[CreditApplicationPlan]:
        """Reserve an economic snapshot without consuming any ledger rows.

        There is deliberately no durable reservation row.  The returned plan
        is an optimistic token: ``apply_settled`` re-locks every selected row
        and compares its balance before it performs any mutation.  Keeping
        this spelling on the public operation makes the provider-before-ledger
        boundary explicit to callers that model the first stage as a reserve.
        """
        return await self.prepare(invoice_id, user_id)

    async def apply_settled(
        self,
        invoice_id: UUID,
        user_id: str,
        plan: CreditApplicationPlan,
        provider_receipt: ProviderReceipt,
        *,
        payment_id: str | None = None,
    ) -> CreditResult[CreditApplication]:
        """Consume a prepared plan only after a matching provider receipt.

        Provider success is a prerequisite, never a side effect of this
        method.  A receipt mismatch leaves the credit ledger untouched.  The
        plan is checked against a fresh locked snapshot before the existing
        transaction-owned application path performs any updates.
        """
        if plan.invoice_id != str(invoice_id) or plan.user_id != user_id:
            return Failure(
                CreditValidationError(message="Credit application plan identity does not match")
            )
        if plan.credit_applied_paisa <= 0:
            return Success(_application(0, plan.gross_paisa, []).unwrap())
        if provider_receipt.amount != plan.credit_applied_paisa:
            return Failure(
                CreditManualReviewError(
                    message="Provider receipt amount does not match credit plan",
                    details={
                        "expected_amount_paisa": plan.credit_applied_paisa,
                        "provider_amount_paisa": provider_receipt.amount,
                        "manual_review": True,
                    },
                    source="credit_apply",
                )
            )
        if payment_id is None or provider_receipt.payment_id != payment_id:
            return Failure(
                CreditManualReviewError(
                    message="Provider receipt payment does not match captured payment",
                    details={
                        "expected_payment_id": payment_id,
                        "provider_payment_id": provider_receipt.payment_id,
                        "manual_review": True,
                    },
                    source="credit_apply",
                )
            )

        session = self._session_factory()
        try:
            async with session, session.begin():
                snapshot = await self._prepare_with_session(session, invoice_id, user_id)
                if isinstance(snapshot, Failure):
                    return snapshot
                current = snapshot.unwrap()
                if current.credit_applied_paisa == 0 and not current.items:
                    # An already committed consumption is a successful
                    # idempotent replay.  A changed, unconsumed snapshot is a
                    # conflict and must not silently refund new rows against
                    # the old provider receipt.
                    existing = await self._has_existing_consumption(session, invoice_id)
                    if existing:
                        return _already_applied_application()
                if not _same_plan(current, plan):
                    return Failure(
                        CreditConflictError(
                            message="Credit application plan is stale",
                            details={"invoice_id": str(invoice_id)},
                            source="credit_apply",
                        )
                    )
                applied = await self._execute_with_session(
                    session,
                    invoice_id,
                    user_id,
                    razorpay_payment_id=provider_receipt.payment_id,
                )
                if isinstance(applied, Failure):
                    raise _CreditApplyAbortError(applied.failure())  # noqa: TRY301
                return applied
        except _CreditApplyAbortError as abort:
            return Failure(abort.error)  # type: ignore[arg-type]

    async def execute(self, invoice_id: UUID, user_id: str) -> CreditResult[CreditApplication]:
        session = self._session_factory()
        try:
            async with session:
                return await self._execute_owned(session, invoice_id, user_id)
        except _CreditApplyAbortError as abort:
            # The transaction context has already performed the rollback.
            return Failure(abort.error)  # type: ignore[arg-type]

    async def _execute_owned(
        self, session: AsyncSession, invoice_id: UUID, user_id: str
    ) -> CreditResult[CreditApplication]:
        async with session.begin():
            result = await self._execute_with_session(session, invoice_id, user_id)
            if isinstance(result, Failure):
                raise _CreditApplyAbortError(result.failure())
            return result

    async def _prepare_with_session(  # noqa: PLR0912, PLR0914
        self, session: AsyncSession, invoice_id: UUID, user_id: str
    ) -> CreditResult[CreditApplicationPlan]:
        credit_repo = self._credit_repo_factory(session)
        consumptions = self._consumptions_factory(session)
        invoices = self._invoices_factory(session)

        invoice_result = await invoices.find_by_id_for_update(invoice_id)
        if isinstance(invoice_result, Failure):
            return invoice_result
        invoice = invoice_result.unwrap()
        if invoice is None:
            return Failure(CreditNotFoundError(message="Invoice not found"))
        if invoice.user_id != user_id:
            return Failure(CreditAuthorizationError(message="Invoice not found"))
        if invoice.status not in {InvoiceStatus.ISSUED.value, InvoiceStatus.PAID.value}:
            return Failure(
                CreditValidationError(
                    message=f"Cannot apply credits to invoice in status '{invoice.status}'"
                )
            )

        gross_paisa = rupees_to_paisa(invoice.total)
        existing_result = await consumptions.find_all_by_invoice_id(invoice_id)
        if isinstance(existing_result, Failure):
            return existing_result
        if existing_result.unwrap():
            return Success(
                CreditApplicationPlan(
                    invoice_id=str(invoice_id),
                    user_id=user_id,
                    gross_paisa=gross_paisa,
                    credit_applied_paisa=0,
                    items=[],
                )
            )

        notes_result = await credit_repo.find_credit_notes_for_consumption(user_id)
        if isinstance(notes_result, Failure):
            return notes_result
        notes = notes_result.unwrap()
        projection_ids = {note.credit_id for note in notes if note.credit_id is not None}
        credits_result = await credit_repo.find_available_for_consumption(user_id)
        if isinstance(credits_result, Failure):
            return credits_result
        all_credits = credits_result.unwrap()
        by_id = {credit.id: credit for credit in all_credits}
        remaining = gross_paisa
        items: list[CreditApplicationItem] = []

        for note in notes:
            if remaining <= 0:
                break
            if note.credit_id is None:
                return Failure(
                    CreditManualReviewError(
                        message="Credit note has no linked ledger projection",
                        details={"credit_note_id": str(note.id), "manual_review": True},
                        source="credit_apply",
                    )
                )
            projection = by_id.get(note.credit_id)
            if projection is None:
                projection_result = await credit_repo.find_by_id(note.credit_id)
                if isinstance(projection_result, Failure):
                    return projection_result
                projection = projection_result.unwrap()
            if projection is None:
                return Failure(
                    CreditManualReviewError(
                        message="Credit note projection is missing",
                        details={"credit_note_id": str(note.id), "manual_review": True},
                        source="credit_apply",
                    )
                )
            amount = min(note.remaining_paisa, remaining)
            if note.remaining_paisa > projection.remaining_balance:
                return Failure(
                    CreditManualReviewError(
                        message="Credit note exceeds its linked ledger projection",
                        details={
                            "credit_note_id": str(note.id),
                            "credit_id": str(projection.id),
                            "note_remaining_paisa": note.remaining_paisa,
                            "projection_remaining_paisa": projection.remaining_balance,
                            "manual_review": True,
                        },
                        source="credit_apply",
                    )
                )
            items.append(
                CreditApplicationItem(
                    credit_id=str(projection.id),
                    credit_note_id=str(note.id),
                    amount_paisa=amount,
                    expected_credit_balance=projection.remaining_balance,
                    expected_note_balance=note.remaining_paisa,
                )
            )
            remaining -= amount

        for credit in all_credits:
            if remaining <= 0:
                break
            if credit.id in projection_ids:
                continue
            amount = min(credit.remaining_balance, remaining)
            items.append(
                CreditApplicationItem(
                    credit_id=str(credit.id),
                    amount_paisa=amount,
                    expected_credit_balance=credit.remaining_balance,
                )
            )
            remaining -= amount
        return Success(
            CreditApplicationPlan(
                invoice_id=str(invoice_id),
                user_id=user_id,
                gross_paisa=gross_paisa,
                credit_applied_paisa=gross_paisa - remaining,
                items=items,
            )
        )

    async def _has_existing_consumption(self, session: AsyncSession, invoice_id: UUID) -> bool:
        result = await self._consumptions_factory(session).find_all_by_invoice_id(invoice_id)
        return isinstance(result, Success) and bool(result.unwrap())

    async def execute_in_session(
        self,
        session: AsyncSession,
        invoice_id: UUID,
        user_id: str,
        *,
        razorpay_payment_id: str | None = None,
    ) -> CreditResult[CreditApplication]:
        """Join an existing invoice/webhook transaction."""
        try:
            result = await self._execute_with_session(
                session,
                invoice_id,
                user_id,
                razorpay_payment_id=razorpay_payment_id,
            )
        except BaseException:
            # Failpoints and cancellation can happen between any two ledger
            # writes.  The joined caller may otherwise catch the exception
            # and commit its outer transaction, so abort it here and then
            # preserve the original signal.
            rollback = getattr(session, "rollback", None)
            if rollback is not None:
                await rollback()
            raise
        if isinstance(result, Failure):
            # A joined call cannot return Failure while its outer caller goes
            # on to commit.  Roll back the complete outer transaction,
            # including writes made before this operation, then translate the
            # internal abort back to the normal Result contract.
            rollback = getattr(session, "rollback", None)
            if rollback is not None:
                await rollback()
        return result

    async def _execute_with_session(  # noqa: PLR0912, PLR0914, PLR0915
        self,
        session: AsyncSession,
        invoice_id: UUID,
        user_id: str,
        *,
        razorpay_payment_id: str | None = None,
    ) -> CreditResult[CreditApplication]:
        credit_repo = self._credit_repo_factory(session)
        consumptions = self._consumptions_factory(session)
        invoices = self._invoices_factory(session)
        audit = self._audit_factory(session)

        invoice_result = await invoices.find_by_id_for_update(invoice_id)
        if isinstance(invoice_result, Failure):
            return invoice_result
        invoice = invoice_result.unwrap()
        if invoice is None:
            return Failure(CreditNotFoundError(message="Invoice not found"))
        if invoice.user_id != user_id:
            return Failure(CreditAuthorizationError(message="Invoice not found"))
        if invoice.status not in {InvoiceStatus.ISSUED.value, InvoiceStatus.PAID.value}:
            return Failure(
                CreditValidationError(
                    message=f"Cannot apply credits to invoice in status '{invoice.status}'"
                )
            )

        gross_paisa = rupees_to_paisa(invoice.total)
        existing_result = await consumptions.find_all_by_invoice_id(invoice_id)
        if isinstance(existing_result, Failure):
            return existing_result
        existing = existing_result.unwrap()
        if existing:
            # The economic application already happened.  A retry is a
            # successful no-op, so report no *new* credit.  The invoice's
            # durable ``credit_applied`` projection is the settlement source
            # on a webhook retry; returning it here would change the public
            # idempotency contract and could make callers display/spend it
            # twice.
            return _already_applied_application()

        notes_result = await credit_repo.find_credit_notes_for_consumption(user_id)
        if isinstance(notes_result, Failure):
            return notes_result
        notes = notes_result.unwrap()
        projection_ids = {note.credit_id for note in notes if note.credit_id is not None}
        credits_result = await credit_repo.find_available_for_consumption(user_id)
        if isinstance(credits_result, Failure):
            return credits_result
        all_credits = credits_result.unwrap()
        standalone_credits = [credit for credit in all_credits if credit.id not in projection_ids]

        remaining = gross_paisa
        applied = 0
        consumed: list[ConsumedCredit] = []
        for note in notes:
            if remaining <= 0:
                break
            if note.credit_id is None:
                return Failure(
                    CreditManualReviewError(
                        message="Credit note has no linked ledger projection",
                        details={"credit_note_id": str(note.id), "manual_review": True},
                        source="credit_apply",
                    )
                )
            projection = next((row for row in all_credits if row.id == note.credit_id), None)
            if projection is None:
                projection_result = await credit_repo.find_by_id(note.credit_id)
                if isinstance(projection_result, Failure):
                    return projection_result
                projection = projection_result.unwrap()
            if projection is None:
                return Failure(
                    CreditManualReviewError(
                        message="Credit note projection is missing",
                        details={"credit_note_id": str(note.id), "manual_review": True},
                        source="credit_apply",
                    )
                )
            amount = min(note.remaining_paisa, remaining)
            if note.remaining_paisa > projection.remaining_balance:
                return Failure(
                    CreditManualReviewError(
                        message="Credit note exceeds its linked ledger projection",
                        details={
                            "credit_note_id": str(note.id),
                            "credit_id": str(projection.id),
                            "note_remaining_paisa": note.remaining_paisa,
                            "projection_remaining_paisa": projection.remaining_balance,
                            "manual_review": True,
                        },
                        source="credit_apply",
                    )
                )
            note_result = await credit_repo.update_credit_note_balance(
                note, new_remaining_paisa=note.remaining_paisa - amount
            )
            if isinstance(note_result, Failure):
                return note_result
            projection_result = await credit_repo.update_balance(
                projection,
                new_remaining_balance=projection.remaining_balance - amount,
                new_status=CreditStatus.CONSUMED
                if projection.remaining_balance - amount == 0
                else None,
                consumed_at=datetime.now(tz=UTC)
                if projection.remaining_balance - amount == 0
                else None,
            )
            if isinstance(projection_result, Failure):
                return projection_result
            application_result = await credit_repo.create_note_application(
                CreditNoteApplication(
                    credit_note_id=note.id, invoice_id=invoice_id, amount_paisa=amount
                )
            )
            if isinstance(application_result, Failure):
                return application_result
            consumption_result = await consumptions.create(
                CreditConsumption(
                    id=uuid4(),
                    user_id=user_id,
                    credit_id=projection.id,
                    invoice_id=invoice_id,
                    consumed_amount=amount,
                    razorpay_payment_id=razorpay_payment_id,
                    metadata_={"credit_note_id": str(note.id)},
                )
            )
            if isinstance(consumption_result, Failure):
                return consumption_result
            consumed.append(
                ConsumedCredit(
                    credit_id=str(projection.id),
                    consumed_amount=amount,
                    remaining_balance=projection.remaining_balance - amount,
                )
            )
            remaining -= amount
            applied += amount

        for credit in standalone_credits:
            if remaining <= 0:
                break
            amount = min(credit.remaining_balance, remaining)
            new_balance = credit.remaining_balance - amount
            updated = await credit_repo.update_balance(
                credit,
                new_remaining_balance=new_balance,
                new_status=CreditStatus.CONSUMED if new_balance == 0 else None,
                consumed_at=datetime.now(tz=UTC) if new_balance == 0 else None,
            )
            if isinstance(updated, Failure):
                return updated
            consumption_result = await consumptions.create(
                CreditConsumption(
                    id=uuid4(),
                    user_id=user_id,
                    credit_id=credit.id,
                    invoice_id=invoice_id,
                    consumed_amount=amount,
                    razorpay_payment_id=razorpay_payment_id,
                    metadata_={},
                )
            )
            if isinstance(consumption_result, Failure):
                return consumption_result
            consumed.append(
                ConsumedCredit(
                    credit_id=str(credit.id), consumed_amount=amount, remaining_balance=new_balance
                )
            )
            remaining -= amount
            applied += amount

        if applied == 0:
            return _application(0, gross_paisa, [])
        cash_due = max(remaining, 0)
        values: dict[str, object] = {"credit_applied": paisa_to_rupees(applied)}
        status = invoice.status
        if cash_due == 0 and status == InvoiceStatus.ISSUED.value:
            status = InvoiceStatus.PAID.value
            values["paid_at"] = datetime.now(tz=UTC)
        invoice_update = await invoices.update_status(invoice, status=status, extra_values=values)
        if isinstance(invoice_update, Failure):
            error = invoice_update.failure()
            return Failure(CreditCollaboratorError(message=error.message, details=error.details))
        audit_result = await audit.create(
            AuditLog(
                entity_type="invoice",
                entity_id=str(invoice_id),
                action="credit.consumed",
                user_id=user_id,
                changes={"credit_applied_paisa": applied, "cash_due_paisa": cash_due},
            )
        )
        if isinstance(audit_result, Failure):
            error = audit_result.failure()
            return Failure(CreditCollaboratorError(message=error.message, details=error.details))
        return _application(applied, gross_paisa, consumed)


def _application(
    applied: int, gross_paisa: int, consumed: list[ConsumedCredit]
) -> Success[CreditApplication]:
    cash_due = max(gross_paisa - applied, 0)
    return Success(
        CreditApplication(
            credit_applied_paisa=applied,
            credit_applied_rupees=applied / 100,
            cash_due_paisa=cash_due,
            cash_due_rupees=cash_due / 100,
            credits_consumed=consumed,
            invoice_paid_in_full=cash_due == 0,
        )
    )


def _already_applied_application() -> Success[CreditApplication]:
    return Success(
        CreditApplication(
            credit_applied_paisa=0,
            credit_applied_rupees=0.0,
            cash_due_paisa=0,
            cash_due_rupees=0.0,
            credits_consumed=[],
            invoice_paid_in_full=True,
        )
    )


def _same_plan(current: CreditApplicationPlan, expected: CreditApplicationPlan) -> bool:
    """Compare only immutable economic facts, independent of list identity."""
    if (
        current.invoice_id != expected.invoice_id
        or current.user_id != expected.user_id
        or current.gross_paisa != expected.gross_paisa
        or current.credit_applied_paisa != expected.credit_applied_paisa
    ):
        return False
    return current.items == expected.items
