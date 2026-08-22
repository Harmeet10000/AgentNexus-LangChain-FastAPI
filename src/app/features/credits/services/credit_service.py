"""Credit service: grant, consume, balance, history, expiration."""

from __future__ import annotations

import calendar
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from returns.result import Failure

from app.features.audit.model import AuditLog
from app.features.credits.dto.consumption_dto import ConsumedCredit, CreditConsumptionResult
from app.features.credits.dto.credit_dto import (
    ConsumptionRecord,
    CreditBalanceResponse,
    CreditGrantDTO,
    CreditGrantResponse,
    CreditHistoryResponse,
    CreditRecord,
)
from app.features.credits.exceptions import (
    CreditAmountMustBePositiveException,
    CreditMetadataMissingException,
)
from app.features.credits.models.consumption import CreditConsumption
from app.features.credits.models.credit import CreditStatus, CreditType, UserCredit
from app.shared.result import app_error_to_exception, log_expected_failure

if TYPE_CHECKING:
    from decimal import Decimal

    from sqlalchemy.ext.asyncio import AsyncSession

    from app.features.audit.repository import AuditLogRepository
    from app.features.credits.repositories.consumption_repository import ConsumptionRepository
    from app.features.credits.repositories.credit_repository import CreditRepository


def _repo_error(error, operation: str) -> None:
    log_expected_failure(error, operation=operation)
    raise app_error_to_exception(error)


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
    ) -> CreditGrantResponse:
        """Grant credit to a user (Requirement 49)."""
        if dto.credit_amount < 1:
            raise CreditAmountMustBePositiveException(dto.credit_amount)

        if dto.credit_type == CreditType.ADMIN_GRANT.value and "admin_user_id" not in dto.metadata_:
            raise CreditMetadataMissingException

        now = datetime.now(tz=UTC)
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

        result = await self.credit_repo.create(credit)
        if isinstance(result, Failure):
            _repo_error(result.failure(), "grant_credit")
        created = result.unwrap()

        await self.audit.create(
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

        return _grant_response(created)

    async def get_credit_balance(self, user_id: str) -> CreditBalanceResponse:
        """Get user's total available credit balance (Requirement 52.1)."""
        result = await self.credit_repo.get_active_balance(user_id)
        if isinstance(result, Failure):
            _repo_error(result.failure(), "get_credit_balance")
        balance_paisa = result.unwrap()
        return CreditBalanceResponse(
            total_balance=balance_paisa,
            total_balance_rupees=balance_paisa / 100,
        )

    async def get_credit_history(
        self,
        user_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> CreditHistoryResponse:
        """Get user's credit and consumption history (Requirement 52.2)."""
        credits_result = await self.credit_repo.find_by_user(user_id, limit=limit, offset=offset)
        if isinstance(credits_result, Failure):
            _repo_error(credits_result.failure(), "get_credit_history")
        credit_rows, credit_total = credits_result.unwrap()

        consumptions_result = await self.consumptions.find_by_user(
            user_id, limit=limit, offset=offset
        )
        if isinstance(consumptions_result, Failure):
            _repo_error(consumptions_result.failure(), "get_credit_history")
        consumption_rows, consumption_total = consumptions_result.unwrap()

        return CreditHistoryResponse(
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

    async def consume_credits(
        self,
        user_id: str,
        invoice_id,
        invoice_gross_total: Decimal,
        session: AsyncSession,  # noqa: ARG002 — owned by caller, required by design
    ) -> CreditConsumptionResult:
        """Apply available credits to an invoice (Requirement 50).

        CRITICAL: This method accepts a session it does NOT own and MUST NOT commit.
        The caller (InvoiceService) owns the transaction boundary.
        """
        available_result = await self.credit_repo.find_available_for_consumption(user_id)
        if isinstance(available_result, Failure):
            _repo_error(available_result.failure(), "consume_credits")
        available_credits = available_result.unwrap()

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
                _repo_error(update_result.failure(), "consume_credits")

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
                _repo_error(consumption_result.failure(), "consume_credits")

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

        await self.audit.create(
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

        return CreditConsumptionResult(
            credit_applied=total_credit_applied_paisa,
            credit_applied_rupees=total_credit_applied_paisa / 100,
            cash_due=cash_due_paisa,
            cash_due_rupees=cash_due_paisa / 100,
            credits_consumed=consumed_credits,
            invoice_paid_in_full=remaining_due <= 0,
        )

    async def expire_credits(self) -> int:
        """Background job to expire past-due credits (Requirement 51). Returns count expired."""
        now = datetime.now(tz=UTC)
        result = await self.credit_repo.expire_credits_past_date(now)
        if isinstance(result, Failure):
            _repo_error(result.failure(), "expire_credits")
        expired = result.unwrap()

        for credit in expired:
            await self.audit.create(
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

        return len(expired)

    async def grant_credit_on_downgrade(
        self,
        user_id: str,
        subscription_id,
        proration_amount_paisa: int,
        *,
        billing_cycle_end: datetime,
    ) -> CreditGrantResponse:
        """Grant credit from plan downgrade proration (Requirement 54)."""
        new_year = billing_cycle_end.year + 1
        new_month = billing_cycle_end.month
        max_day = calendar.monthrange(new_year, new_month)[1]
        valid_until = billing_cycle_end.replace(
            year=new_year,
            month=new_month,
            day=min(billing_cycle_end.day, max_day),
        )

        dto = CreditGrantDTO(
            credit_type=CreditType.PLAN_CREDIT.value,
            credit_amount=proration_amount_paisa,
            valid_from=datetime.now(tz=UTC),
            valid_until=valid_until,
            description=f"Proration credit for subscription {subscription_id}",
            metadata_={"subscription_id": str(subscription_id)},
        )
        return await self.grant_credit(user_id, dto)
