"""Adversarial contracts for the billing money and retry boundaries.

These tests intentionally stop at public feature seams.  The doubles model the
transaction/remote-provider boundaries, while the real coordinator, webhook
service, dunning route and credit application command remain under test.  They
are deliberately stricter than happy-path characterization tests: a retry
must not turn a remote mutation into two mutations, and a financial ledger
must never silently manufacture value.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from fastapi import Response
from returns.result import Failure, Success

from app.features.billing.credits.apply import ApplyCreditsToInvoice
from app.features.billing.credits.models import CreditStatus
from app.features.billing.credits.service import CreditService
from app.features.billing.dunning.router import run_dunning_now
from app.features.billing.providers import (
    CreatePaymentLinkOperation,
    CreateRefundOperation,
    CreateSubscriptionOperation,
    InMemoryBillingProvider,
    ProviderOperationKind,
    ProviderOutcomeUnknownError,
    ProviderPaymentLink,
    ProviderReceipt,
)
from app.features.billing.subscriptions.coordinator import (
    CoordinatorStatus,
    InMemoryLifecyclePersistence,
    LifecycleRecord,
    SubscriptionLifecycleCoordinator,
)
from app.features.billing.subscriptions.lifecycle import (
    ActiveState,
    ChangeSubscriptionPlan,
    CommandMetadata,
    EventMetadata,
    PaymentCaptured,
    PolicyContext,
    RemoteSyncState,
    SubscriptionSnapshot,
)
from app.features.billing.subscriptions.models import SubscriptionStatus
from app.features.billing.webhooks.models import WebhookEventStatus, WebhookEventType
from app.features.billing.webhooks.service import WebhookService

if TYPE_CHECKING:
    from typing import Any
    from uuid import UUID

    from app.features.billing.credits.models import CreditNote


NOW = datetime(2026, 10, 20, 12, tzinfo=UTC)
PERIOD_END = datetime(2026, 11, 1, tzinfo=UTC)


def _active_snapshot(
    *, subscription_id: str = "sub-local", version: int = 4
) -> SubscriptionSnapshot:
    return SubscriptionSnapshot(
        id=subscription_id,
        user_id="user-1",
        plan_id="plan-old",
        state=ActiveState(datetime(2026, 10, 1, tzinfo=UTC), PERIOD_END),
        version=version,
        remote_sync=RemoteSyncState.SYNCED,
        observed_plan_id="plan-old",
    )


def _coordinator() -> tuple[
    SubscriptionLifecycleCoordinator, InMemoryLifecyclePersistence, InMemoryBillingProvider
]:
    provider = InMemoryBillingProvider()
    persistence = InMemoryLifecyclePersistence()
    persistence.seed(
        LifecycleRecord(
            _active_snapshot(),
            provider_customer_id="cust-1",
            provider_subscription_id="sub-remote",
            provider_plan_id="rp_plan_old",
            desired_provider_plan_id="rp_plan_new",
        )
    )
    return SubscriptionLifecycleCoordinator(persistence, provider), persistence, provider


@pytest.mark.asyncio
async def test_upgrade_exposes_payment_link_but_gates_remote_plan_and_entitlement_until_capture() -> (
    None
):
    """An upgrade is desired locally before it is paid, not remotely active."""
    coordinator, persistence, provider = _coordinator()
    command = ChangeSubscriptionPlan(
        CommandMetadata("upgrade-gate-1", expected_version=4),
        "plan-new",
        NOW,
    )

    outcome = await coordinator.apply(
        command,
        "upgrade-gate-1",
        subscription_id="sub-local",
        context=PolicyContext(
            now=NOW,
            current_plan_amount=1_000,
            target_plan_amount=2_000,
            current_interval="monthly",
            target_interval="monthly",
            target_currency="INR",
            target_provider_plan_id="rp_plan_new",
        ),
    )

    assert isinstance(outcome, Success)
    assert outcome.unwrap().snapshot.plan_id == "plan-new"
    assert outcome.unwrap().snapshot.remote_sync is RemoteSyncState.PENDING
    assert outcome.unwrap().snapshot.grants_paid_entitlement is False
    assert [call.operation.kind for call in provider.calls] == [
        ProviderOperationKind.CREATE_PAYMENT_LINK
    ]
    link = next(
        result.value
        for result in outcome.unwrap().provider_results
        if isinstance(result.value, ProviderPaymentLink)
    )
    assert link.short_url == "https://example.test/pay/1"
    assert persistence.records["sub-local"].provider_plan_id == "rp_plan_old"


@pytest.mark.asyncio
async def test_verified_capture_executes_pending_remote_plan_change_once() -> None:
    coordinator, persistence, provider = _coordinator()
    seeded = await provider.execute(
        CreateSubscriptionOperation("rp_plan_old", "cust-1", 0, 1, True, "seed-capture")
    )
    assert isinstance(seeded, Success)
    persistence.records["sub-local"] = replace(
        persistence.records["sub-local"],
        provider_subscription_id=seeded.unwrap().value.provider_subscription_id,
    )
    upgrade = ChangeSubscriptionPlan(
        CommandMetadata("upgrade-capture-1", expected_version=4), "plan-new", NOW
    )
    first = await coordinator.apply(
        upgrade,
        "upgrade-capture-1",
        subscription_id="sub-local",
        context=PolicyContext(
            now=NOW,
            current_plan_amount=1_000,
            target_plan_amount=2_000,
            current_interval="monthly",
            target_interval="monthly",
            target_currency="INR",
            target_provider_plan_id="rp_plan_new",
        ),
    )
    assert isinstance(first, Success)
    pending_upgrade = first.unwrap().snapshot.pending_upgrade
    assert pending_upgrade is not None

    capture = PaymentCaptured(
        EventMetadata("capture-upgrade-1", occurred_at=NOW),
        payment_id="pay-upgrade-1",
        amount_paisa=pending_upgrade.amount_paisa,
        currency=pending_upgrade.currency,
    )
    captured = await coordinator.observe(
        capture,
        "capture-upgrade-1",
        subscription_id="sub-local",
        context=PolicyContext(now=NOW),
    )
    duplicate = await coordinator.observe(
        capture,
        "capture-upgrade-1",
        subscription_id="sub-local",
        context=PolicyContext(now=NOW),
    )

    assert isinstance(captured, Success)
    assert captured.unwrap().status in {CoordinatorStatus.APPLIED, CoordinatorStatus.RECONCILED}
    assert isinstance(duplicate, Success)
    assert [call.operation.kind for call in provider.calls].count(
        ProviderOperationKind.CHANGE_SUBSCRIPTION_PLAN
    ) == 1
    assert persistence.records["sub-local"].provider_plan_id == "rp_plan_new"


@pytest.mark.asyncio
async def test_provider_idempotency_same_payload_replays_and_different_payload_conflicts() -> None:
    coordinator, _persistence, provider = _coordinator()
    first_operation = CreatePaymentLinkOperation(500, "INR", "upgrade", "payment-key-1")
    replay = await coordinator.execute_provider_operation(
        first_operation, subscription_id="sub-local", idempotency_key="payment-key-1"
    )
    again = await coordinator.execute_provider_operation(
        first_operation, subscription_id="sub-local", idempotency_key="payment-key-1"
    )
    conflict = await coordinator.execute_provider_operation(
        CreatePaymentLinkOperation(700, "INR", "upgrade", "payment-key-other"),
        subscription_id="sub-local",
        idempotency_key="payment-key-1",
    )

    assert isinstance(replay, Success)
    assert isinstance(again, Success)
    assert again.unwrap() == replay.unwrap()
    assert isinstance(conflict, Failure)
    assert [call.operation.kind for call in provider.calls].count(
        ProviderOperationKind.CREATE_PAYMENT_LINK
    ) == 1


@pytest.mark.asyncio
async def test_provider_idempotency_concurrent_replays_issue_one_remote_mutation() -> None:
    coordinator, _persistence, provider = _coordinator()
    operation = CreatePaymentLinkOperation(500, "INR", "upgrade", "payment-concurrent-1")

    results = await asyncio.gather(
        *(
            coordinator.execute_provider_operation(
                operation,
                subscription_id="sub-local",
                idempotency_key="payment-concurrent-1",
            )
            for _ in range(2)
        )
    )

    assert all(isinstance(result, Success) for result in results)
    assert results[0].unwrap() == results[1].unwrap()
    assert [call.operation.kind for call in provider.calls].count(
        ProviderOperationKind.CREATE_PAYMENT_LINK
    ) == 1


class _UnknownThenFoundProvider(InMemoryBillingProvider):
    """Provider double that simulates a crash after the remote mutation."""

    def __init__(self) -> None:
        super().__init__()
        self.mutation_attempts = 0
        self.receipt = ProviderReceipt("rfnd-recovered", "pay-1", 500)

    async def execute(self, operation: Any):  # type: ignore[no-untyped-def]
        if operation.kind is ProviderOperationKind.CREATE_REFUND:
            self.mutation_attempts += 1
            if self.mutation_attempts == 1:
                return Failure(
                    ProviderOutcomeUnknownError(operation.kind, operation.correlation_id)
                )
        if operation.kind is ProviderOperationKind.FIND_REFUND_BY_CORRELATION:
            return Success(
                SimpleNamespace(
                    operation=operation.kind,
                    correlation_id=operation.correlation_id,
                    value=self.receipt,
                )
            )
        return await super().execute(operation)


@pytest.mark.asyncio
async def test_keyless_cancel_refund_recovery_reconciles_without_second_refund() -> None:
    provider = _UnknownThenFoundProvider()
    persistence = InMemoryLifecyclePersistence()
    coordinator = SubscriptionLifecycleCoordinator(persistence, provider)
    operation = CreateRefundOperation("pay-1", 500, "cancel-refund-recovery")

    first = await coordinator.execute_provider_operation(
        operation,
        subscription_id=None,
        idempotency_key="cancel-refund-recovery",
    )
    second = await coordinator.execute_provider_operation(
        operation,
        subscription_id=None,
        idempotency_key="cancel-refund-recovery",
    )

    assert isinstance(first, Success)
    assert isinstance(second, Success)
    assert first.unwrap().value == second.unwrap().value == provider.receipt
    assert provider.mutation_attempts == 1


@pytest.mark.asyncio
async def test_concurrent_pending_refund_lease_reconciles_without_second_mutation() -> None:
    """A retry racing the first sender must read the correlation, never send twice."""

    class _YieldingProvider(InMemoryBillingProvider):
        def __init__(self) -> None:
            super().__init__()
            self.first_send = asyncio.Event()
            self.first_read = asyncio.Event()
            self.release = asyncio.Event()
            self.refund_mutations = 0

        async def execute(self, operation: Any):  # type: ignore[no-untyped-def]
            if operation.kind is ProviderOperationKind.CREATE_REFUND:
                self.refund_mutations += 1
                self.first_send.set()
                await self.release.wait()
            elif operation.kind is ProviderOperationKind.FIND_REFUND_BY_CORRELATION:
                self.first_read.set()
                await self.release.wait()
            return await super().execute(operation)

    provider = _YieldingProvider()
    persistence = InMemoryLifecyclePersistence()
    coordinator = SubscriptionLifecycleCoordinator(persistence, provider)
    operation = CreateRefundOperation("pay-1", 500, "concurrent-refund")

    first = asyncio.create_task(
        coordinator.execute_provider_operation(
            operation, subscription_id=None, idempotency_key="concurrent-refund"
        )
    )
    await provider.first_send.wait()
    second = asyncio.create_task(
        coordinator.execute_provider_operation(
            operation, subscription_id=None, idempotency_key="concurrent-refund"
        )
    )
    await provider.first_read.wait()
    provider.release.set()
    first_result, second_result = await asyncio.gather(first, second)

    assert isinstance(first_result, Success)
    assert isinstance(second_result, Success)
    assert provider.refund_mutations == 1


class _WebhookRepo:
    def __init__(self, event: object) -> None:
        self.event = event
        self.statuses: list[str] = []

    async def find_by_razorpay_event_id(self, _event_id: str):
        return Success(self.event)

    async def update_status(
        self, event: object, *, status: str, extra_values: dict[str, object] | None = None
    ):
        del extra_values
        self.statuses.append(status)
        event.status = status
        return Success(event)


@pytest.mark.asyncio
async def test_fresh_processing_webhook_duplicate_is_successful_noop() -> None:
    event = SimpleNamespace(
        id="event-processing",
        status=WebhookEventStatus.PROCESSING.value,
        event_type=WebhookEventType.PAYMENT_CAPTURED.value,
        payload={},
        retry_count=0,
        updated_at=datetime.now(tz=UTC),
    )
    repo = _WebhookRepo(event)
    service = WebhookService(
        webhooks=repo,  # type: ignore[arg-type]
        subscriptions=SimpleNamespace(),
        plans=SimpleNamespace(),
        audit=SimpleNamespace(),
        payment_service=SimpleNamespace(),
        invoice_service=SimpleNamespace(),
    )

    result = await service.process(
        event_id="event-processing",
        event_type=WebhookEventType.PAYMENT_CAPTURED.value,
        payload={},
    )

    assert isinstance(result, Success)
    assert result.unwrap() is True
    assert repo.statuses == []


@pytest.mark.asyncio
async def test_dunning_route_exposes_retry_payment_link_url() -> None:
    subscription = SimpleNamespace(
        id=uuid4(),
        status=SubscriptionStatus.PAST_DUE.value,
        payment_link_url="https://example.test/dunning/1",
    )

    async def find_due_for_retry(**_: object):
        return Success([subscription])

    async def execute_retry(_subscription: object):
        return Success(subscription)

    service = SimpleNamespace(find_due_for_retry=find_due_for_retry, execute_retry=execute_retry)
    response = Response()

    rendered = await run_dunning_now(service, response, limit=1)  # type: ignore[arg-type]

    assert rendered.data["items"] == [
        {
            "subscription_id": str(subscription.id),
            "status": SubscriptionStatus.PAST_DUE.value,
            "payment_link_url": "https://example.test/dunning/1",
        }
    ]


class _TransactionSession:
    def __init__(self) -> None:
        self.commits = 0
        self.rollbacks = 0

    async def __aenter__(self) -> _TransactionSession:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc_type is None:
            self.commits += 1
        else:
            self.rollbacks += 1

    def begin(self) -> _TransactionSession:
        return self


class _CreditRepo:
    def __init__(self, credits: list[SimpleNamespace]) -> None:
        self.credits = credits

    async def find_available_for_consumption(self, _user_id: str):
        return Success([credit for credit in self.credits if credit.remaining_balance > 0])

    async def find_credit_notes_for_consumption(self, _user_id: str):
        return Success([])

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


class _ConsumptionRepo:
    def __init__(self) -> None:
        self.rows: list[object] = []

    async def create(self, row: object):
        self.rows.append(row)
        return Success(row)

    async def find_all_by_invoice_id(self, _invoice_id: UUID):
        return Success(list(self.rows))


class _InvoiceRepo:
    def __init__(self, invoice: SimpleNamespace) -> None:
        self.invoice = invoice
        self.updates = 0

    async def find_by_id_for_update(self, _invoice_id: UUID):
        return Success(self.invoice)

    async def update_status(
        self,
        invoice: SimpleNamespace,
        *,
        status: str,
        extra_values: dict[str, object] | None = None,
    ):
        invoice.status = status
        self.updates += 1
        if extra_values:
            for key, value in extra_values.items():
                setattr(invoice, key, value)
        return Success(invoice)


@pytest.mark.asyncio
async def test_repeated_invoice_application_is_idempotent() -> None:
    invoice = SimpleNamespace(id=uuid4(), user_id="user-1", status="issued", total=Decimal("10.00"))
    credit = SimpleNamespace(
        id=uuid4(),
        user_id="user-1",
        remaining_balance=1_000,
        status=CreditStatus.ACTIVE.value,
    )
    sessions = [_TransactionSession(), _TransactionSession()]
    credit_repo = _CreditRepo([credit])
    consumption_repo = _ConsumptionRepo()
    invoice_repo = _InvoiceRepo(invoice)

    # The command expects an async-session factory; this wrapper gives it the
    # same object for both the session and transaction context managers.
    class _Factory:
        def __call__(self) -> _TransactionSession:
            return sessions.pop(0)

    command = ApplyCreditsToInvoice(
        _Factory(),
        lambda _session: credit_repo,  # type: ignore[arg-type]
        lambda _session: consumption_repo,  # type: ignore[arg-type]
        lambda _session: invoice_repo,  # type: ignore[arg-type]
        lambda _session: SimpleNamespace(create=_success_none),  # type: ignore[arg-type]
    )

    first = await command.execute(invoice.id, "user-1")
    second = await command.execute(invoice.id, "user-1")

    assert isinstance(first, Success)
    assert isinstance(second, Success)
    assert first.unwrap().invoice_paid_in_full is True
    assert second.unwrap().invoice_paid_in_full is True
    assert second.unwrap().credit_applied_paisa == 0
    assert second.unwrap().cash_due_paisa == 0
    assert len(consumption_repo.rows) == 1
    assert invoice_repo.updates == 1


@pytest.mark.asyncio
async def test_downgrade_credit_requires_source_invoice_and_links_credit_note_to_user_credit() -> (
    None
):
    async def execute(_statement: object) -> SimpleNamespace:
        return _db_result(None)

    session = SimpleNamespace(
        execute=execute,
        flush=_async_none,
    )
    note_rows: list[CreditNote] = []
    credit_rows: list[SimpleNamespace] = []
    credit_repo = SimpleNamespace(
        find_credit_note_by_operation_key=lambda _key: _success_async(None),
        create_credit_note=lambda note: _record_note(note, note_rows),
        create=lambda credit: _record_credit(credit, credit_rows),
        find_by_id=lambda _credit_id: _success_async(credit_rows[-1] if credit_rows else None),
        find_credit_notes_for_consumption=lambda _user_id: _success_async([]),
    )
    audit = SimpleNamespace(create=lambda _audit: _success_none())
    service = CreditService(session, credit_repo, SimpleNamespace(), audit)  # type: ignore[arg-type]

    result = await service.grant_credit_on_downgrade(
        "user-1",
        uuid4(),
        500,
        billing_cycle_end=PERIOD_END,
        currency="INR",
        operation_key="downgrade-note-1",
    )

    assert isinstance(result, Failure)
    assert note_rows == []


@pytest.mark.asyncio
async def test_credit_note_and_user_credit_share_operation_and_amount() -> None:
    invoice_id = uuid4()

    async def execute(_statement: object) -> SimpleNamespace:
        return _db_result(invoice_id)

    session = SimpleNamespace(
        execute=execute,
        flush=_async_none,
    )
    note_rows: list[CreditNote] = []
    credit_rows: list[Any] = []
    credit_repo = SimpleNamespace(
        find_credit_note_by_operation_key=lambda _key: _success_async(None),
        create_credit_note=lambda note: _record_note(note, note_rows),
        create=lambda credit: _record_credit(credit, credit_rows),
        find_by_id=lambda _credit_id: _success_async(credit_rows[-1] if credit_rows else None),
    )
    service = CreditService(
        session,
        credit_repo,
        SimpleNamespace(),
        SimpleNamespace(create=lambda _audit: _success_none()),
    )  # type: ignore[arg-type]

    result = await service.grant_credit_on_downgrade(
        "user-1",
        uuid4(),
        500,
        billing_cycle_end=PERIOD_END,
        currency="INR",
        operation_key="downgrade-note-2",
        source_invoice_id=invoice_id,
    )

    assert isinstance(result, Success)
    assert len(note_rows) == 1
    assert len(credit_rows) == 1
    assert note_rows[0].source_invoice_id == invoice_id
    assert note_rows[0].amount_paisa == note_rows[0].remaining_paisa == credit_rows[0].credit_amount
    assert note_rows[0].metadata_["credit_id"] == str(credit_rows[0].id)


def _db_result(value: object) -> SimpleNamespace:
    return SimpleNamespace(scalar_one_or_none=lambda: value)


async def _async_none() -> None:
    return None


def _success(value: object):
    return Success(value)


async def _success_async(value: object):
    return Success(value)


async def _success_none(_value: object = None) -> Success[None]:
    return Success(None)


async def _record_note(note: CreditNote, rows: list[CreditNote]):
    rows.append(note)
    return Success(note)


async def _record_credit(credit: Any, rows: list[Any]):
    if getattr(credit, "created_at", None) is None:
        credit.created_at = NOW
    rows.append(credit)
    return Success(credit)
