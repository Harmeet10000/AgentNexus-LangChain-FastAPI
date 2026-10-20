"""Second-pass adversarial contracts for the billing ledger.

These tests intentionally use the public service/repository/job seams.  They
are aimed at failures that happy-path tests routinely miss: a renewal payment
must apply the same credit ledger as an initial payment, a long note ledger
must not be truncated, and every spend must be compare-and-set guarded.
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from returns.result import Failure, Success
from sqlalchemy.exc import IntegrityError

from app.features.billing.credits.apply import ApplyCreditsToInvoice
from app.features.billing.credits.models import CreditNoteApplication, CreditStatus
from app.features.billing.credits.repository import CreditRepository
from app.features.billing.invoices.repository import InvoiceRepository
from app.features.billing.subscriptions.lifecycle import TransitionOutcome
from app.features.billing.subscriptions.service import SubscriptionService
from app.features.billing.webhooks.models import WebhookEventType
from app.features.billing.webhooks.service import WebhookService
from database.base import Base
from database.registry import MODEL_MODULE_NAMES, register_models

NOW = datetime(2026, 10, 20, 12, tzinfo=UTC)


def _db_result(value: object) -> SimpleNamespace:
    return SimpleNamespace(scalar_one_or_none=lambda: value)


def test_credit_note_application_is_immutable_per_note_and_invoice() -> None:
    constraints = {constraint.name for constraint in CreditNoteApplication.__table__.constraints}
    assert "uq_credit_note_applications_note_invoice" in constraints
    assert any(name.endswith("ck_credit_note_applications_amount_positive") for name in constraints)
    assert CreditNoteApplication.__table__.c.credit_note_id.nullable is False
    assert CreditNoteApplication.__table__.c.invoice_id.nullable is False


def test_billing_models_are_registered_before_metadata_is_inspected() -> None:
    modules = register_models()
    assert "app.features.billing.credits.models" in MODEL_MODULE_NAMES
    assert len(modules) == len(MODEL_MODULE_NAMES)
    assert {
        "credit_notes",
        "user_credits",
        "credit_note_applications",
    } <= set(Base.metadata.tables)


def test_credit_ledger_migration_keeps_projection_and_application_invariants() -> None:
    migration = (
        Path(__file__).parents[3] / "src/alembic/versions/0025_credit_ledger_authority.py"
    ).read_text(encoding="utf-8")
    assert 'revision: str = "0025"' in migration
    assert 'down_revision: str | tuple[str, ...] | None = "0024"' in migration
    assert '"credit_note_applications"' in migration
    assert '"uq_credit_note_applications_note_invoice"' in migration
    assert '"ck_credit_note_applications_amount_positive"' in migration
    assert '"credit_id"' in migration


def test_credit_repository_balance_update_is_compare_and_set() -> None:
    source = inspect.getsource(CreditRepository.update_balance)
    assert "UserCredit.id == credit.id" in source
    assert "UserCredit.remaining_balance == credit.remaining_balance" in source
    assert "UserCredit.remaining_balance >= new_remaining_balance" in source
    assert "Credit balance changed concurrently" in source


def test_credit_repository_note_update_is_compare_and_set() -> None:
    source = inspect.getsource(CreditRepository.update_credit_note_balance)
    assert "CreditNote.id == note.id" in source
    assert "CreditNote.remaining_paisa == note.remaining_paisa" in source
    assert 'CreditNote.status == "available"' in source


@pytest.mark.asyncio
async def test_credit_note_application_conflict_rolls_back_before_returning_failure() -> None:
    class _FailingSession(_Session):
        async def flush(self) -> None:
            statement = "insert"
            raise IntegrityError(statement, {}, RuntimeError("duplicate"))

        async def rollback(self) -> None:
            self.rollbacks += 1

    session = _FailingSession()
    repository = CreditRepository(session)  # type: ignore[arg-type]
    result = await repository.create_note_application(
        SimpleNamespace(credit_note_id=uuid4(), invoice_id=uuid4())  # type: ignore[arg-type]
    )

    assert isinstance(result, Failure)
    assert session.rollbacks == 1


class _Session:
    def __init__(self) -> None:
        self.added: list[object] = []
        self.flushed = 0
        self.rollbacks = 0

    def add(self, value: object) -> None:
        self.added.append(value)

    async def flush(self) -> None:
        self.flushed += 1


class _CreditRepoForNotes:
    def __init__(self, notes: list[SimpleNamespace], projections: list[SimpleNamespace]) -> None:
        self.notes = notes
        self.projections = projections
        self.applications: list[CreditNoteApplication] = []

    async def find_credit_notes_for_consumption(self, _user_id: str):
        return Success(self.notes)

    async def find_available_for_consumption(self, _user_id: str):
        return Success(self.projections)

    async def update_credit_note_balance(self, note: SimpleNamespace, *, new_remaining_paisa: int):
        note.remaining_paisa = new_remaining_paisa
        note.status = "applied" if new_remaining_paisa == 0 else "available"
        return Success(note)

    async def update_balance(
        self,
        credit: SimpleNamespace,
        *,
        new_remaining_balance: int,
        new_status: object = None,
        consumed_at: datetime | None = None,
    ):
        credit.remaining_balance = new_remaining_balance
        if new_status is not None:
            credit.status = getattr(new_status, "value", new_status)
        credit.consumed_at = consumed_at
        return Success(credit)

    async def create_note_application(self, application: CreditNoteApplication):
        self.applications.append(application)
        return Success(application)


class _ConsumptionRepo:
    def __init__(self) -> None:
        self.rows: list[object] = []

    async def find_all_by_invoice_id(self, _invoice_id: object):
        return Success(self.rows)

    async def create(self, row: object):
        self.rows.append(row)
        return Success(row)


class _InvoiceRepo:
    def __init__(self, invoice: SimpleNamespace) -> None:
        self.invoice = invoice

    async def find_by_id_for_update(self, _invoice_id: object):
        return Success(self.invoice)

    async def update_status(self, invoice: SimpleNamespace, *, status: str, extra_values=None):
        invoice.status = status
        if extra_values:
            for key, value in extra_values.items():
                setattr(invoice, key, value)
        return Success(invoice)


class _AuditRepo:
    async def create(self, _entry: object):
        return Success(None)


@pytest.mark.asyncio
async def test_authoritative_credit_notes_are_not_truncated_after_one_hundred() -> None:
    invoice = SimpleNamespace(id=uuid4(), user_id="user-1", status="issued", total=Decimal("1.01"))
    notes: list[SimpleNamespace] = []
    projections: list[SimpleNamespace] = []
    for _ in range(101):
        credit_id = uuid4()
        notes.append(
            SimpleNamespace(
                id=uuid4(),
                credit_id=credit_id,
                remaining_paisa=1,
                amount_paisa=1,
                status="available",
            )
        )
        projections.append(
            SimpleNamespace(
                id=credit_id,
                remaining_balance=1,
                status=CreditStatus.ACTIVE.value,
            )
        )
    credit_repo = _CreditRepoForNotes(notes, projections)
    consumption_repo = _ConsumptionRepo()
    operation = ApplyCreditsToInvoice(
        session_factory=object(),  # type: ignore[arg-type]
        credit_repo_factory=lambda _session: credit_repo,  # type: ignore[return-value]
        consumptions_factory=lambda _session: consumption_repo,  # type: ignore[return-value]
        invoices_factory=lambda _session: _InvoiceRepo(invoice),  # type: ignore[return-value]
        audit_factory=lambda _session: _AuditRepo(),  # type: ignore[return-value]
    )

    result = await operation.execute_in_session(
        _Session(),
        invoice.id,
        "user-1",  # type: ignore[arg-type]
    )

    assert isinstance(result, Success)
    assert result.unwrap().credit_applied_paisa == 101
    assert result.unwrap().cash_due_paisa == 0
    assert len(credit_repo.applications) == 101
    assert len(consumption_repo.rows) == 101


class _WebhookRepo:
    def __init__(self) -> None:
        self.event: object | None = None
        self.statuses: list[str] = []

    async def find_by_razorpay_event_id(self, _event_id: str):
        return Success(self.event)

    async def create(self, event: object):
        self.event = event
        return Success(event)

    async def update_status(self, event: object, *, status: str, extra_values=None):
        del extra_values
        self.statuses.append(status)
        event.status = status
        return Success(event)


class _InvoiceRepoDouble(InvoiceRepository):
    def __init__(self, invoice: object) -> None:
        self.invoice = invoice

    async def find_by_payment_id(self, _payment_id: object):
        return Success(self.invoice)


@pytest.mark.asyncio
async def test_payment_captured_renewal_applies_credits_once_through_public_webhook_service() -> (
    None
):
    subscription = SimpleNamespace(id=uuid4(), user_id="user-1", plan_id=uuid4(), version=4)
    payment = SimpleNamespace(id=uuid4())
    invoice = SimpleNamespace(id=uuid4())
    credit_apply = SimpleNamespace(execute_in_session=AsyncMock(return_value=Success(None)))
    invoice_repository = _InvoiceRepoDouble(invoice)
    invoice_service = SimpleNamespace(
        invoices=invoice_repository,
        session=SimpleNamespace(
            execute=AsyncMock(return_value=_db_result(None)),
        ),
        generate_for_payment=AsyncMock(return_value=Success(invoice)),
        generate_receipt_for_payment=AsyncMock(return_value=Success(None)),
    )
    lifecycle = SimpleNamespace(
        observe=AsyncMock(
            return_value=Success(
                SimpleNamespace(plan=SimpleNamespace(outcome=TransitionOutcome.APPLY))
            )
        )
    )
    service = WebhookService(
        webhooks=_WebhookRepo(),  # type: ignore[arg-type]
        subscriptions=SimpleNamespace(
            find_by_razorpay_id=AsyncMock(return_value=Success(subscription))
        ),
        plans=SimpleNamespace(
            find_by_id=AsyncMock(return_value=Success(SimpleNamespace(id=uuid4())))
        ),
        audit=SimpleNamespace(create=AsyncMock(return_value=Success(None))),
        payment_service=SimpleNamespace(
            record_payment=AsyncMock(return_value=Success(payment)),
        ),
        invoice_service=invoice_service,  # type: ignore[arg-type]
        lifecycle_coordinator=lifecycle,  # type: ignore[arg-type]
        credit_apply=credit_apply,  # type: ignore[arg-type]
    )

    payload = {
        "payment": {
            "entity": {
                "id": "pay_renewal_1",
                "subscription_id": "sub_renewal_1",
                "amount": 1_000,
                "currency": "INR",
            }
        }
    }
    first = await service.process(
        event_id="evt-renewal-1",
        event_type=WebhookEventType.PAYMENT_CAPTURED.value,
        payload=payload,
    )
    duplicate = await service.process(
        event_id="evt-renewal-1",
        event_type=WebhookEventType.PAYMENT_CAPTURED.value,
        payload=payload,
    )

    assert first == Success(True)
    assert duplicate == Success(True)
    credit_apply.execute_in_session.assert_awaited_once_with(
        invoice_service.session, invoice.id, subscription.user_id
    )


@pytest.mark.asyncio
async def test_subscription_downgrade_passes_captured_payment_invoice_as_source() -> None:
    subscription = SimpleNamespace(
        id=uuid4(),
        user_id="user-1",
        plan_id=uuid4(),
        status="active",
        version=4,
        current_period_start=datetime(2026, 10, 1, tzinfo=UTC),
        current_period_end=datetime(2026, 11, 1, tzinfo=UTC),
        metadata_={},
        razorpay_subscription_id="sub_remote",
        currency_display="INR",
        remote_sync="synced",
        observed_plan_id=None,
        entitlement_until=None,
        cancel_at_period_end=False,
        pause_start=None,
        pause_end=None,
        retry_count=0,
        trial_end=None,
        cancellation_requested_at=None,
        created_at=NOW,
        updated_at=NOW,
    )
    # Reassign after the explicit field list above to keep the fixture obvious
    # while satisfying the model-shaped response contract.
    subscription.current_period_end = datetime(2026, 11, 1, tzinfo=UTC)
    current = SimpleNamespace(
        id=subscription.plan_id,
        name="Pro",
        amount=2_000,
        interval="monthly",
        currency="INR",
        tax_rate=Decimal("0.18"),
        razorpay_plan_id="rp_old",
        is_active=True,
    )
    target = SimpleNamespace(
        id=uuid4(),
        name="Starter",
        amount=1_000,
        interval="monthly",
        currency="INR",
        tax_rate=Decimal("0.18"),
        razorpay_plan_id="rp_new",
        is_active=True,
    )
    invoice_id = uuid4()
    credit_service = SimpleNamespace(
        grant_credit_on_downgrade=AsyncMock(return_value=Success(SimpleNamespace()))
    )
    coordinator = SimpleNamespace(
        apply=AsyncMock(return_value=Success(SimpleNamespace(provider_results=())))
    )
    subscriptions = SimpleNamespace(
        find_by_id=AsyncMock(side_effect=[Success(subscription), Success(subscription)])
    )
    plans = SimpleNamespace(find_by_id=AsyncMock(side_effect=[Success(current), Success(target)]))
    payments = SimpleNamespace(
        find_by_subscription=AsyncMock(
            return_value=Success([SimpleNamespace(invoice_id=invoice_id, status="captured")])
        )
    )
    service = SubscriptionService(
        object(),  # type: ignore[arg-type]
        subscriptions,  # type: ignore[arg-type]
        plans,  # type: ignore[arg-type]
        payments,  # type: ignore[arg-type]
        SimpleNamespace(create=AsyncMock(return_value=Success(None))),  # type: ignore[arg-type]
        SimpleNamespace(),  # type: ignore[arg-type]
        coordinator,  # type: ignore[arg-type]
        credit_service,  # type: ignore[arg-type]
    )

    result = await service.change_plan(
        "user-1",
        str(subscription.id),
        SimpleNamespace(new_plan_id=str(target.id), effective_date=NOW),
    )

    assert isinstance(result, Success)
    credit_service.grant_credit_on_downgrade.assert_awaited_once()
    assert (
        credit_service.grant_credit_on_downgrade.await_args.kwargs["source_invoice_id"]
        == invoice_id
    )
