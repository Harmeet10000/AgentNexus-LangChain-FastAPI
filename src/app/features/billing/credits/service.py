"""Credit service: grant, consume, balance, history, expiration."""

from __future__ import annotations

import calendar
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from returns.result import Failure, Success

from app.features.audit.models import AuditLog
from app.features.billing.credits.exceptions import (
    CreditAmountError,
    CreditCollaboratorError,
    CreditManualReviewError,
    CreditMetadataError,
    CreditValidationError,
)
from app.features.billing.credits.models import (
    CreditConsumption,
    CreditNote,
    CreditStatus,
    CreditType,
    UserCredit,
)

from .dto import (
    ConsumedCredit,
    ConsumptionRecord,
    CreditBalanceResponse,
    CreditConsumptionResult,
    CreditGrantDTO,
    CreditGrantResponse,
    CreditHistoryResponse,
    CreditRecord,
)

if TYPE_CHECKING:
    from decimal import Decimal

    from sqlalchemy.ext.asyncio import AsyncSession

    from app.features.audit.repository import AuditLogRepository
    from app.features.billing.credits.repository import (
        ConsumptionRepository,
        CreditRepository,
    )

    from .exceptions import CreditResult


def _grant_response(credit: UserCredit) -> CreditGrantResponse:
    return CreditGrantResponse(
        credit_id=str(credit.id),
        user_id=credit.user_id,
        credit_type=credit.credit_type,
        credit_amount=credit.credit_amount,
        remaining_balance=credit.remaining_balance,
        valid_from=credit.valid_from,
        valid_until=credit.valid_until,
        status=credit.status,
        created_at=credit.created_at,
    )


class CreditService:
    """Core credit business logic."""

    def __init__(
        self,
        session: AsyncSession,
        credit_repo: CreditRepository,
        consumptions: ConsumptionRepository,
        audit: AuditLogRepository,
    ) -> None:
        self.session = session
        self.credit_repo = credit_repo
        self.consumptions = consumptions
        self.audit = audit

    async def grant_credit(
        self,
        user_id: str,
        dto: CreditGrantDTO,
    ) -> CreditResult[CreditGrantResponse]:
        """Grant credit to a user (Requirement 49)."""
        if dto.credit_amount < 1:
            return Failure(
                CreditAmountError(
                    message=f"Credit amount must be positive, got {dto.credit_amount}"
                )
            )

        if dto.credit_type == CreditType.ADMIN_GRANT.value and "admin_user_id" not in dto.metadata_:
            return Failure(
                CreditMetadataError(message="ADMIN_GRANT requires admin_user_id in metadata")
            )

        now: datetime = datetime.now(tz=UTC)
        credit = UserCredit(
            id=uuid4(),
            user_id=user_id,
            credit_type=dto.credit_type,
            credit_amount=dto.credit_amount,
            remaining_balance=dto.credit_amount,
            valid_from=dto.valid_from or now,
            valid_until=dto.valid_until,
            status=CreditStatus.ACTIVE.value,
            metadata_=dto.metadata_,
        )

        result: CreditResult[UserCredit] = await self.credit_repo.create(credit)
        if isinstance(result, Failure):
            return result
        created = result.unwrap()

        audit_result = await self.audit.create(
            AuditLog(
                entity_type="user_credit",
                entity_id=str(created.id),
                action="credit.granted",
                user_id=user_id,
                changes={
                    "credit_type": created.credit_type,
                    "credit_amount": created.credit_amount,
                },
            )
        )

        if isinstance(audit_result, Failure):
            error = audit_result.failure()
            return Failure(CreditCollaboratorError(message=error.message, details=error.details))
        return Success(_grant_response(created))

    async def get_credit_balance(self, user_id: str) -> CreditResult[CreditBalanceResponse]:
        """Get user's total available credit balance (Requirement 52.1)."""
        result = await self.credit_repo.get_active_balance(user_id)
        if isinstance(result, Failure):
            return result
        balance_paisa = result.unwrap()
        return Success(
            CreditBalanceResponse(
                total_balance=balance_paisa,
                total_balance_rupees=balance_paisa / 100,
            )
        )

    async def get_credit_history(
        self,
        user_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> CreditResult[CreditHistoryResponse]:
        """Get user's credit and consumption history (Requirement 52.2)."""
        credits_result = await self.credit_repo.find_by_user(user_id, limit=limit, offset=offset)
        if isinstance(credits_result, Failure):
            return credits_result
        credit_rows, credit_total = credits_result.unwrap()

        consumptions_result = await self.consumptions.find_by_user(
            user_id, limit=limit, offset=offset
        )
        if isinstance(consumptions_result, Failure):
            return consumptions_result
        consumption_rows, consumption_total = consumptions_result.unwrap()

        return Success(
            CreditHistoryResponse(
                credits=[
                    CreditRecord(
                        credit_id=str(c.id),
                        credit_type=c.credit_type,
                        credit_amount=c.credit_amount,
                        remaining_balance=c.remaining_balance,
                        valid_from=c.valid_from,
                        valid_until=c.valid_until,
                        status=c.status,
                        created_at=c.created_at,
                    )
                    for c in credit_rows
                ],
                consumptions=[
                    ConsumptionRecord(
                        consumption_id=str(co.id),
                        credit_id=str(co.credit_id),
                        consumed_amount=co.consumed_amount,
                        invoice_id=str(co.invoice_id) if co.invoice_id else None,
                        razorpay_payment_id=co.razorpay_payment_id,
                        created_at=co.created_at,
                    )
                    for co in consumption_rows
                ],
                total=credit_total + consumption_total,
                limit=limit,
                offset=offset,
            )
        )

    async def consume_credits(
        self,
        user_id: str,
        invoice_id,
        invoice_gross_total: Decimal,
        session: AsyncSession,  # noqa: ARG002 — owned by caller, required by design
    ) -> CreditResult[CreditConsumptionResult]:
        """Apply available credits to an invoice (Requirement 50).

        CRITICAL: This method accepts a session it does NOT own and MUST NOT commit.
        The caller (InvoiceService) owns the transaction boundary.
        """
        available_result = await self.credit_repo.find_available_for_consumption(user_id)
        if isinstance(available_result, Failure):
            return available_result
        available_credits = available_result.unwrap()

        if (invoice_gross_total * 100) % 1 != 0:
            return Failure(
                CreditValidationError(
                    message="Invoice total has fractional paisa",
                    details={"invoice_gross_total": str(invoice_gross_total)},
                )
            )
        total_due_paisa = int(invoice_gross_total * 100)
        remaining_due = total_due_paisa
        consumed_credits: list[ConsumedCredit] = []
        total_credit_applied_paisa = 0

        for credit in available_credits:
            if remaining_due <= 0:
                break

            consume_amount = min(credit.remaining_balance, remaining_due)
            new_balance = credit.remaining_balance - consume_amount
            is_fully_consumed = new_balance == 0

            update_result = await self.credit_repo.update_balance(
                credit,
                new_remaining_balance=new_balance,
                new_status=CreditStatus.CONSUMED if is_fully_consumed else None,
                consumed_at=datetime.now(tz=UTC) if is_fully_consumed else None,
            )
            if isinstance(update_result, Failure):
                return update_result

            consumption = CreditConsumption(
                id=uuid4(),
                user_id=user_id,
                credit_id=credit.id,
                invoice_id=invoice_id,
                consumed_amount=consume_amount,
                metadata_={},
            )
            consumption_result = await self.consumptions.create(consumption)
            if isinstance(consumption_result, Failure):
                return consumption_result

            consumed_credits.append(
                ConsumedCredit(
                    credit_id=str(credit.id),
                    consumed_amount=consume_amount,
                    remaining_balance=new_balance,
                )
            )

            remaining_due -= consume_amount
            total_credit_applied_paisa += consume_amount

        cash_due_paisa = max(remaining_due, 0)

        audit_result = await self.audit.create(
            AuditLog(
                entity_type="user_credit",
                entity_id=user_id,
                action="credit.consumed",
                user_id=user_id,
                changes={
                    "invoice_id": str(invoice_id),
                    "credit_applied_paisa": total_credit_applied_paisa,
                    "cash_due_paisa": cash_due_paisa,
                },
            )
        )

        if isinstance(audit_result, Failure):
            error = audit_result.failure()
            return Failure(CreditCollaboratorError(message=error.message, details=error.details))
        return Success(
            CreditConsumptionResult(
                credit_applied=total_credit_applied_paisa,
                credit_applied_rupees=total_credit_applied_paisa / 100,
                cash_due=cash_due_paisa,
                cash_due_rupees=cash_due_paisa / 100,
                credits_consumed=consumed_credits,
                invoice_paid_in_full=remaining_due <= 0,
            )
        )

    async def expire_credits(self) -> CreditResult[int]:
        """Background job to expire past-due credits (Requirement 51). Returns count expired."""
        now = datetime.now(tz=UTC)
        result = await self.credit_repo.expire_credits_past_date(now)
        if isinstance(result, Failure):
            return result
        expired = result.unwrap()

        for credit in expired:
            audit_result = await self.audit.create(
                AuditLog(
                    entity_type="user_credit",
                    entity_id=str(credit.id),
                    action="credit.expired",
                    user_id=credit.user_id,
                    changes={
                        "credit_id": str(credit.id),
                        "credit_amount": credit.credit_amount,
                    },
                )
            )
            if isinstance(audit_result, Failure):
                error = audit_result.failure()
                return Failure(
                    CreditCollaboratorError(message=error.message, details=error.details)
                )

        return Success(len(expired))

    async def grant_credit_on_downgrade(  # noqa: PLR0912, PLR0914
        self,
        user_id: str,
        subscription_id,
        proration_amount_paisa: int,
        *,
        billing_cycle_end: datetime,
        currency: str | None = None,
        operation_key: str | None = None,
        source_invoice_id: UUID | None = None,
    ) -> CreditResult[CreditGrantResponse]:
        """Grant credit from plan downgrade proration (Requirement 54)."""
        if operation_key is None:
            return Failure(
                CreditMetadataError(message="Downgrade credit requires an operation key")
            )
        if currency is None:
            return Failure(CreditMetadataError(message="Downgrade credit requires plan currency"))
        if source_invoice_id is None:
            # The original invoice is the GST/audit anchor.  Looking up "the
            # latest invoice" is unsafe under concurrent renewals and can
            # silently attach a credit to the wrong economic event.
            return Failure(
                CreditManualReviewError(
                    message="Downgrade credit requires the authoritative source invoice",
                    details={
                        "operation_key": operation_key,
                        "subscription_id": str(subscription_id),
                        "manual_review": True,
                    },
                    source="credit_service",
                )
            )
        existing_note_method = getattr(self.credit_repo, "find_credit_note_by_operation_key", None)
        if existing_note_method is not None:
            existing_note = await existing_note_method(operation_key)
            if isinstance(existing_note, Failure):
                return existing_note
            note = existing_note.unwrap()
            if note is not None:
                credit_id = note.credit_id
                if credit_id is None:
                    metadata_credit_id = note.metadata_.get("credit_id")
                    credit_id = (
                        UUID(metadata_credit_id) if isinstance(metadata_credit_id, str) else None
                    )
                existing_credit = None
                if isinstance(credit_id, UUID):
                    existing_credit = await self.credit_repo.find_by_id(credit_id)
                if existing_credit is None:
                    find_credit = getattr(self.credit_repo, "find_by_operation_key", None)
                    if find_credit is not None:
                        existing_credit = await find_credit(operation_key)
                if isinstance(existing_credit, Success):
                    existing_credit_row = existing_credit.unwrap()
                    if existing_credit_row is not None:
                        return Success(_grant_response(existing_credit_row))
        new_year = billing_cycle_end.year + 1
        new_month = billing_cycle_end.month
        max_day = calendar.monthrange(new_year, new_month)[1]
        valid_until = billing_cycle_end.replace(
            year=new_year,
            month=new_month,
            day=min(billing_cycle_end.day, max_day),
        )

        credit_id = uuid4()
        now = datetime.now(tz=UTC)
        note = CreditNote(
            id=uuid4(),
            credit_id=credit_id,
            user_id=user_id,
            subscription_id=subscription_id,
            source_invoice_id=source_invoice_id,
            operation_key=operation_key,
            amount_paisa=proration_amount_paisa,
            remaining_paisa=proration_amount_paisa,
            currency=currency,
            applies_at=billing_cycle_end,
            # Assign a fresh dict before the first flush.  In-place mutation of
            # a plain JSONB dict after flush is not reliably dirty-tracked.
            metadata_={"credit_id": str(credit_id)},
        )
        dto = CreditGrantDTO(
            credit_type=CreditType.PLAN_CREDIT.value,
            credit_amount=proration_amount_paisa,
            valid_from=now,
            valid_until=valid_until,
            description=f"Proration credit for subscription {subscription_id}",
            metadata_={
                "subscription_id": str(subscription_id),
                "operation_key": operation_key,
            },
        )
        credit = UserCredit(
            id=credit_id,
            user_id=user_id,
            credit_type=dto.credit_type,
            credit_amount=dto.credit_amount,
            remaining_balance=dto.credit_amount,
            valid_from=now,
            valid_until=valid_until,
            status=CreditStatus.ACTIVE.value,
            metadata_={
                **dto.metadata_,
                "operation_key": operation_key,
                "credit_note_id": str(note.id),
            },
        )
        credit_result = await self.credit_repo.create(credit)
        if isinstance(credit_result, Failure):
            return credit_result
        note_result = await self.credit_repo.create_credit_note(note)
        if isinstance(note_result, Failure):
            return note_result
        audit_result = await self.audit.create(
            AuditLog(
                entity_type="user_credit",
                entity_id=str(credit.id),
                action="credit.granted",
                user_id=user_id,
                changes={
                    "credit_type": credit.credit_type,
                    "credit_amount": credit.credit_amount,
                    "credit_note_id": str(note.id),
                    "source_invoice_id": str(source_invoice_id),
                },
            )
        )
        if isinstance(audit_result, Failure):
            error = audit_result.failure()
            return Failure(CreditCollaboratorError(message=error.message, details=error.details))
        return Success(_grant_response(credit))
