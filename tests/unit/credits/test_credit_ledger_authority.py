"""Focused contracts for the authoritative downgrade-credit ledger."""

from __future__ import annotations

import inspect
from decimal import Decimal
from types import SimpleNamespace
from typing import get_args, get_type_hints
from uuid import uuid4

from returns.result import Success

from app.features.billing.credits.apply import ApplyCreditsToInvoice
from app.features.billing.credits.dependencies import (
    get_apply_credits_operation,
    get_db_session_local,
)
from app.features.billing.credits.models import CreditNote, UserCredit
from app.features.billing.credits.repository import CreditRepository
from app.features.billing.invoices.models import InvoiceStatus


def test_credit_models_allow_multiple_active_rows_and_match_note_checks() -> None:
    user_constraints = {constraint.name for constraint in UserCredit.__table__.constraints}
    note_constraints = {constraint.name for constraint in CreditNote.__table__.constraints}

    assert "uq_user_credits_user_id_active" not in {
        index.name for index in UserCredit.__table__.indexes
    }
    assert any(
        name.endswith("ck_user_credits_remaining_balance_valid") for name in user_constraints
    )
    assert any(name.endswith("ck_credit_notes_remaining_valid") for name in note_constraints)
    assert any(name.endswith("ck_credit_notes_status") for name in note_constraints)


def test_credit_repository_does_not_truncate_available_credits() -> None:
    signature = inspect.signature(CreditRepository.find_available_for_consumption)
    assert signature.parameters["limit"].default is None


def test_apply_dependency_uses_session_factory_not_request_session() -> None:
    annotation = get_type_hints(get_apply_credits_operation, include_extras=True)["session_factory"]
    dependency_marker = get_args(annotation)[1]
    assert dependency_marker.dependency is get_db_session_local


def _credit(remaining: int) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(),
        remaining_balance=remaining,
        status="active",
        user_id="user-1",
    )


class _CreditRepository:
    def __init__(self, credits: list[SimpleNamespace]) -> None:
        self.credits = credits

    async def find_credit_notes_for_consumption(self, user_id: str) -> Success[list[object]]:
        del user_id
        return Success([])

    async def find_available_for_consumption(self, user_id: str) -> Success[list[SimpleNamespace]]:
        del user_id
        return Success(self.credits)

    async def update_balance(
        self, credit: SimpleNamespace, **kwargs: object
    ) -> Success[SimpleNamespace]:
        credit.remaining_balance = kwargs["new_remaining_balance"]
        return Success(credit)


class _ConsumptionRepository:
    def __init__(self) -> None:
        self.rows: list[object] = []

    async def find_all_by_invoice_id(self, invoice_id: object) -> Success[list[object]]:
        del invoice_id
        return Success(self.rows)

    async def create(self, row: object) -> Success[object]:
        self.rows.append(row)
        return Success(row)


class _InvoiceRepository:
    def __init__(self) -> None:
        self.invoice = SimpleNamespace(
            id=uuid4(), user_id="user-1", status=InvoiceStatus.ISSUED.value, total=Decimal("1.01")
        )

    async def find_by_id_for_update(self, invoice_id: object) -> Success[object]:
        del invoice_id
        return Success(self.invoice)

    async def update_status(self, invoice: object, **kwargs: object) -> Success[object]:
        invoice.status = kwargs["status"]
        return Success(invoice)


class _AuditRepository:
    async def create(self, row: object) -> Success[object]:
        return Success(row)


def test_apply_consumes_more_than_one_hundred_standalone_credits() -> None:
    credit_rows = [_credit(1) for _ in range(101)]
    credit_repo = _CreditRepository(credit_rows)
    consumption_repo = _ConsumptionRepository()
    invoice_repo = _InvoiceRepository()
    operation = ApplyCreditsToInvoice(
        session_factory=object(),  # type: ignore[arg-type]
        credit_repo_factory=lambda _session: credit_repo,  # type: ignore[return-value]
        consumptions_factory=lambda _session: consumption_repo,  # type: ignore[return-value]
        invoices_factory=lambda _session: invoice_repo,  # type: ignore[return-value]
        audit_factory=lambda _session: _AuditRepository(),  # type: ignore[return-value]
    )

    import asyncio

    result = asyncio.run(operation.execute_in_session(object(), invoice_repo.invoice.id, "user-1"))  # type: ignore[arg-type]

    assert result.unwrap().credit_applied_paisa == 101
    assert len(consumption_repo.rows) == 101
