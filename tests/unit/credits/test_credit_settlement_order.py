"""Contracts for the provider-before-credit settlement boundary."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from returns.result import Failure, Success

from app.features.billing.credits.apply import (
    ApplyCreditsToInvoice,
    CreditApplicationItem,
    CreditApplicationPlan,
)
from app.features.billing.invoices.repository import InvoiceRepository
from app.features.billing.payments.exceptions import PaymentProviderError
from app.features.billing.payments.models import PaymentStatus
from app.features.billing.payments.service import PaymentService
from app.features.billing.providers import (
    InMemoryBillingProvider,
    ProviderOperationKind,
    ProviderReceipt,
)
from app.features.billing.subscriptions.coordinator import (
    InMemoryLifecyclePersistence,
    LifecycleRecord,
    SubscriptionLifecycleCoordinator,
)
from app.features.billing.subscriptions.lifecycle import ActiveState, SubscriptionSnapshot
from app.features.billing.webhooks.service import WebhookService


def _plan(invoice_id: object, user_id: str = "user-1") -> CreditApplicationPlan:
    return CreditApplicationPlan(
        invoice_id=str(invoice_id),
        user_id=user_id,
        gross_paisa=500,
        credit_applied_paisa=300,
        items=[
            CreditApplicationItem(
                credit_id=str(uuid4()),
                amount_paisa=300,
                expected_credit_balance=300,
            )
        ],
    )


@pytest.mark.asyncio
async def test_apply_settled_rejects_mismatched_receipt_before_opening_transaction() -> None:
    session_factory = AsyncMock(
        side_effect=AssertionError("provider receipt must be checked first")
    )
    operation = ApplyCreditsToInvoice(
        session_factory,  # type: ignore[arg-type]
        lambda _session: SimpleNamespace(),  # type: ignore[arg-type]
        lambda _session: SimpleNamespace(),  # type: ignore[arg-type]
        lambda _session: SimpleNamespace(),  # type: ignore[arg-type]
        lambda _session: SimpleNamespace(),  # type: ignore[arg-type]
    )
    invoice_id = uuid4()

    result = await operation.apply_settled(
        invoice_id,
        "user-1",
        _plan(invoice_id),
        ProviderReceipt(provider_refund_id="rfnd-1", payment_id="pay-1", amount=299),
        payment_id="pay-1",
    )

    assert isinstance(result, Failure)
    session_factory.assert_not_called()


@pytest.mark.asyncio
async def test_webhook_settles_provider_before_local_credit_apply() -> None:
    invoice_id = uuid4()
    invoice = SimpleNamespace(id=invoice_id)
    payment = SimpleNamespace(id=uuid4(), razorpay_payment_id="pay-1")
    subscription = SimpleNamespace(id=uuid4(), user_id="user-1")
    plan = _plan(invoice_id)
    order: list[str] = []

    class _Invoices(InvoiceRepository):
        async def find_by_payment_id(self, _payment_id: object):
            return Success(invoice)

    credit_apply = SimpleNamespace(
        prepare=AsyncMock(return_value=Success(plan)),
        apply_settled=AsyncMock(
            side_effect=lambda *_args, **_kwargs: order.append("apply") or Success(None)
        ),
    )
    payment_service = SimpleNamespace(
        settle_credit_application=AsyncMock(
            side_effect=lambda *_args, **_kwargs: (
                order.append("provider")
                or Success(
                    ProviderReceipt(provider_refund_id="rfnd-1", payment_id="pay-1", amount=300)
                )
            )
        )
    )
    service = WebhookService(
        webhooks=SimpleNamespace(),  # type: ignore[arg-type]
        subscriptions=SimpleNamespace(),  # type: ignore[arg-type]
        plans=SimpleNamespace(),  # type: ignore[arg-type]
        audit=SimpleNamespace(),  # type: ignore[arg-type]
        payment_service=payment_service,  # type: ignore[arg-type]
        invoice_service=SimpleNamespace(invoices=_Invoices(object()), session=object()),  # type: ignore[arg-type]
        credit_apply=credit_apply,  # type: ignore[arg-type]
    )

    result = await service._settle_captured_credit(payment, subscription)

    assert result == Success(None)
    assert order == ["provider", "apply"]
    credit_apply.apply_settled.assert_awaited_once()


@pytest.mark.asyncio
async def test_webhook_provider_failure_does_not_apply_credit() -> None:
    invoice_id = uuid4()
    invoice = SimpleNamespace(id=invoice_id)
    payment = SimpleNamespace(id=uuid4(), razorpay_payment_id="pay-1")
    subscription = SimpleNamespace(id=uuid4(), user_id="user-1")
    credit_apply = SimpleNamespace(
        prepare=AsyncMock(return_value=Success(_plan(invoice_id))),
        apply_settled=AsyncMock(),
    )
    payment_service = SimpleNamespace(
        settle_credit_application=AsyncMock(
            return_value=Failure(PaymentProviderError(message="provider unavailable"))
        )
    )

    class _Invoices(InvoiceRepository):
        async def find_by_payment_id(self, _payment_id: object):
            return Success(invoice)

    service = WebhookService(
        webhooks=SimpleNamespace(),  # type: ignore[arg-type]
        subscriptions=SimpleNamespace(),  # type: ignore[arg-type]
        plans=SimpleNamespace(),  # type: ignore[arg-type]
        audit=SimpleNamespace(),  # type: ignore[arg-type]
        payment_service=payment_service,  # type: ignore[arg-type]
        invoice_service=SimpleNamespace(invoices=_Invoices(object()), session=object()),  # type: ignore[arg-type]
        credit_apply=credit_apply,  # type: ignore[arg-type]
    )

    result = await service._settle_captured_credit(payment, subscription)

    assert isinstance(result, Failure)
    credit_apply.apply_settled.assert_not_awaited()


@pytest.mark.asyncio
async def test_webhook_unknown_provider_outcome_does_not_consume_credit() -> None:
    """An ambiguous refund is not permission to consume the prepared plan."""
    invoice_id = uuid4()
    invoice = SimpleNamespace(id=invoice_id)
    payment = SimpleNamespace(id=uuid4(), razorpay_payment_id="pay-unknown")
    subscription = SimpleNamespace(id=uuid4(), user_id="user-1")
    credit_apply = SimpleNamespace(
        prepare=AsyncMock(return_value=Success(_plan(invoice_id))),
        apply_settled=AsyncMock(),
    )
    payment_service = SimpleNamespace(
        settle_credit_application=AsyncMock(
            return_value=Failure(
                PaymentProviderError(
                    message="provider refund outcome is unknown; reconciliation required"
                )
            )
        )
    )

    class _Invoices(InvoiceRepository):
        async def find_by_payment_id(self, _payment_id: object):
            return Success(invoice)

    service = WebhookService(
        webhooks=SimpleNamespace(),  # type: ignore[arg-type]
        subscriptions=SimpleNamespace(),  # type: ignore[arg-type]
        plans=SimpleNamespace(),  # type: ignore[arg-type]
        audit=SimpleNamespace(),  # type: ignore[arg-type]
        payment_service=payment_service,  # type: ignore[arg-type]
        invoice_service=SimpleNamespace(invoices=_Invoices(object()), session=object()),  # type: ignore[arg-type]
        credit_apply=credit_apply,  # type: ignore[arg-type]
    )

    result = await service._settle_captured_credit(payment, subscription)

    assert isinstance(result, Failure)
    credit_apply.apply_settled.assert_not_awaited()


class _CrashAfterRefundReceipt(InMemoryBillingProvider):
    """Raise after the provider has durably produced a refund receipt."""

    def __init__(self) -> None:
        super().__init__()
        self.refund_posts = 0
        self._crash = True

    async def execute(self, operation):  # type: ignore[no-untyped-def]
        result = await super().execute(operation)
        if operation.kind is ProviderOperationKind.CREATE_REFUND:
            self.refund_posts += 1
            if self._crash:
                self._crash = False
                message = "worker crashed after receiving refund receipt"
                raise RuntimeError(message)
        return result


@pytest.mark.asyncio
async def test_webhook_retry_after_receipt_crash_applies_credit_once() -> None:
    """A receipt survives the crash and is reconciled without a second refund."""
    invoice_id = uuid4()
    invoice = SimpleNamespace(id=invoice_id)
    payment = SimpleNamespace(
        id=uuid4(),
        subscription_id="sub-receipt-crash",
        razorpay_payment_id="pay-receipt-crash",
        amount=1_000,
        currency="INR",
        status=PaymentStatus.CAPTURED.value,
        refund_amount=Decimal(0),
        metadata_={},
    )
    subscription = SimpleNamespace(id=uuid4(), user_id="user-1")
    plan = _plan(invoice_id)

    class _Payments:
        async def find_by_id_for_update(self, _payment_id: str):
            return Success(payment)

        async def update_status(self, _payment: object, *, status: str, extra_values=None):
            payment.status = status
            for key, value in (extra_values or {}).items():
                setattr(payment, key, value)
            return Success(payment)

    persistence = InMemoryLifecyclePersistence(
        {
            "sub-receipt-crash": LifecycleRecord(
                snapshot=SubscriptionSnapshot(
                    id="sub-receipt-crash",
                    user_id="user-1",
                    plan_id="plan-local",
                    state=ActiveState(
                        period_start=datetime(2026, 10, 1, tzinfo=UTC),
                        period_end=datetime(2026, 11, 1, tzinfo=UTC),
                    ),
                    version=1,
                ),
                provider_subscription_id="sub-provider",
                provider_plan_id="plan-provider",
            )
        }
    )
    provider = _CrashAfterRefundReceipt()
    coordinator = SubscriptionLifecycleCoordinator(persistence, provider)
    payment_service = PaymentService(
        payments=_Payments(),  # type: ignore[arg-type]
        audit=SimpleNamespace(create=AsyncMock(return_value=Success(None))),  # type: ignore[arg-type]
        lifecycle_coordinator=coordinator,  # type: ignore[arg-type]
    )
    apply_settled = AsyncMock(
        return_value=Success(None),
    )
    credit_apply = SimpleNamespace(
        prepare=AsyncMock(return_value=Success(plan)),
        apply_settled=apply_settled,
    )

    class _Invoices(InvoiceRepository):
        async def find_by_payment_id(self, _payment_id: object):
            return Success(invoice)

    service = WebhookService(
        webhooks=SimpleNamespace(),  # type: ignore[arg-type]
        subscriptions=SimpleNamespace(),  # type: ignore[arg-type]
        plans=SimpleNamespace(),  # type: ignore[arg-type]
        audit=SimpleNamespace(),  # type: ignore[arg-type]
        payment_service=payment_service,  # type: ignore[arg-type]
        invoice_service=SimpleNamespace(invoices=_Invoices(object()), session=object()),  # type: ignore[arg-type]
        credit_apply=credit_apply,  # type: ignore[arg-type]
    )

    first = await service._settle_captured_credit(payment, subscription)
    second = await service._settle_captured_credit(payment, subscription)

    assert isinstance(first, Failure)
    assert isinstance(second, Success)
    assert apply_settled.await_count == 1
    assert provider.refund_posts == 1
    assert (
        sum(
            call.operation.kind is ProviderOperationKind.FIND_REFUND_BY_CORRELATION
            for call in provider.calls
        )
        == 1
    )


@pytest.mark.asyncio
async def test_settled_consumption_records_the_verified_payment_projection() -> None:
    """The local ledger keeps the provider payment identity atomically."""
    invoice_id = uuid4()
    credit = SimpleNamespace(
        id=uuid4(),
        remaining_balance=500,
        status="active",
        user_id="user-1",
    )
    invoice = SimpleNamespace(
        id=invoice_id,
        user_id="user-1",
        status="issued",
        total=Decimal("5.00"),
    )
    consumptions: list[object] = []

    class _Credits:
        async def find_credit_notes_for_consumption(self, _user_id: str):
            return Success([])

        async def find_available_for_consumption(self, _user_id: str):
            return Success([credit])

        async def update_balance(self, row: object, **kwargs: object):
            row.remaining_balance = kwargs["new_remaining_balance"]
            return Success(row)

    class _Consumption:
        async def find_all_by_invoice_id(self, _invoice_id: object):
            return Success(consumptions)

        async def create(self, row: object):
            consumptions.append(row)
            return Success(row)

    class _Invoices:
        async def find_by_id_for_update(self, _invoice_id: object):
            return Success(invoice)

        async def update_status(self, row: object, **kwargs: object):
            row.status = kwargs["status"]
            return Success(row)

    class _Audit:
        async def create(self, row: object):
            return Success(row)

    operation = ApplyCreditsToInvoice(
        session_factory=object(),  # type: ignore[arg-type]
        credit_repo_factory=lambda _session: _Credits(),  # type: ignore[return-value]
        consumptions_factory=lambda _session: _Consumption(),  # type: ignore[return-value]
        invoices_factory=lambda _session: _Invoices(),  # type: ignore[return-value]
        audit_factory=lambda _session: _Audit(),  # type: ignore[return-value]
    )

    result = await operation.execute_in_session(  # type: ignore[arg-type]
        object(), invoice_id, "user-1", razorpay_payment_id="pay-captured"
    )

    assert isinstance(result, Success)
    assert len(consumptions) == 1
    assert consumptions[0].razorpay_payment_id == "pay-captured"


@pytest.mark.asyncio
async def test_local_credit_failpoint_rolls_back_joined_transaction() -> None:
    """A crash between ledger writes cannot escape into an outer commit."""
    credit = SimpleNamespace(
        id=uuid4(),
        remaining_balance=500,
        status="active",
        user_id="user-1",
    )
    initial_balance = credit.remaining_balance
    session = SimpleNamespace()

    async def rollback() -> None:
        credit.remaining_balance = initial_balance

    session.rollback = AsyncMock(side_effect=rollback)
    consumptions: list[object] = []

    class _Credits:
        async def find_credit_notes_for_consumption(self, _user_id: str):
            return Success([])

        async def find_available_for_consumption(self, _user_id: str):
            return Success([credit])

        async def update_balance(self, row: object, **kwargs: object):
            row.remaining_balance = kwargs["new_remaining_balance"]
            message = "failpoint after balance CAS"
            raise RuntimeError(message)

    class _Consumption:
        async def find_all_by_invoice_id(self, _invoice_id: object):
            return Success(consumptions)

    class _Invoices:
        async def find_by_id_for_update(self, _invoice_id: object):
            return Success(
                SimpleNamespace(
                    id=_invoice_id,
                    user_id="user-1",
                    status="issued",
                    total=Decimal("5.00"),
                )
            )

    operation = ApplyCreditsToInvoice(
        session_factory=object(),  # type: ignore[arg-type]
        credit_repo_factory=lambda _session: _Credits(),  # type: ignore[return-value]
        consumptions_factory=lambda _session: _Consumption(),  # type: ignore[return-value]
        invoices_factory=lambda _session: _Invoices(),  # type: ignore[return-value]
        audit_factory=lambda _session: SimpleNamespace(),  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError, match="failpoint"):
        await operation.execute_in_session(session, uuid4(), "user-1")  # type: ignore[arg-type]

    session.rollback.assert_awaited_once()
    assert credit.remaining_balance == initial_balance
    assert consumptions == []
