"""Independent adversarial contracts for the final billing invariants.

The tests in this module deliberately sit on the public coordinator/service
ports and use failpoint doubles at the provider or repository boundary.  They
model the crashes that happen between two durable writes, where a happy-path
test cannot prove that a retry is safe.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from returns.result import Failure, Success
from sqlalchemy.dialects import postgresql

from app.features.billing.payments.models import PaymentStatus
from app.features.billing.payments.repository import PaymentRepository
from app.features.billing.payments.service import PaymentService
from app.features.billing.providers import (
    CreatePaymentLinkOperation,
    CreateRefundOperation,
    CreateSubscriptionOperation,
    InMemoryBillingProvider,
    ProviderOperationKind,
    ProviderPaymentLink,
    ProviderPaymentState,
    ProviderReceipt,
    ProviderSubscriptionState,
    ProviderSuccess,
    ProviderUnavailableError,
)
from app.features.billing.subscriptions.coordinator import (
    CapturedUpgrade,
    InMemoryLifecyclePersistence,
    IntentStatus,
    LifecycleCoordinatorTransientProviderError,
    LifecycleRecord,
    LifecycleRequest,
    ProviderOperationIntent,
    SubscriptionLifecycleCoordinator,
)
from app.features.billing.subscriptions.dto import SubscriptionCancelDTO
from app.features.billing.subscriptions.exceptions import (
    SubscriptionInfrastructureError,
    SubscriptionRefundError,
)
from app.features.billing.subscriptions.lifecycle import (
    ActiveState,
    ChangeSubscriptionPlan,
    CommandMetadata,
    CreatedState,
    CreateSubscription,
    PolicyContext,
    SubscriptionSnapshot,
)
from app.features.billing.subscriptions.models import (
    ProviderOperationRecord,
    ProviderOperationState,
    SubscriptionStatus,
)
from app.features.billing.subscriptions.repository import SqlAlchemyLifecyclePersistence
from app.features.billing.subscriptions.service import SubscriptionService
from app.features.billing.webhooks.service import WebhookService

if TYPE_CHECKING:
    from typing import Any


NOW = datetime(2026, 10, 20, 12, tzinfo=UTC)
PLAN_LOCAL = "plan-local-uuid"
PLAN_PROVIDER = "rp_plan_bound"


def _created_record() -> LifecycleRecord:
    return LifecycleRecord(
        snapshot=SubscriptionSnapshot(
            id="sub-child-crash",
            user_id="user-1",
            plan_id=PLAN_LOCAL,
            state=CreatedState(),
            version=0,
        ),
        customer_email="customer@example.test",
        desired_provider_plan_id=PLAN_PROVIDER,
    )


class _CrashAfterChildPostProvider(InMemoryBillingProvider):
    """The first child POST succeeds remotely, then the worker crashes."""

    def __init__(self) -> None:
        super().__init__()
        self.subscription_posts = 0

    async def execute(self, operation: Any):  # type: ignore[no-untyped-def]
        if operation.kind is ProviderOperationKind.CREATE_SUBSCRIPTION:
            self.subscription_posts += 1
            if self.subscription_posts == 1:
                message = "worker crashed after child POST"
                raise RuntimeError(message)
            return Success(
                ProviderSuccess(
                    operation.kind,
                    operation.correlation_id,
                    ProviderSubscriptionState(
                        provider_subscription_id="sub-remote-1",
                        status="active",
                        plan_id=operation.plan_id,
                    ),
                )
            )
        if operation.kind is ProviderOperationKind.FIND_SUBSCRIPTION_BY_CORRELATION:
            return Success(
                ProviderSuccess(
                    operation.kind,
                    operation.correlation_id,
                    ProviderSubscriptionState(
                        provider_subscription_id="sub-remote-1",
                        status="active",
                        plan_id=PLAN_PROVIDER,
                    ),
                )
            )
        return await super().execute(operation)


@pytest.mark.asyncio
async def test_customer_child_intent_crash_is_replayable_without_a_second_subscription_post() -> (
    None
):
    """A crash between child-intent persistence and its POST must be safe."""
    persistence = InMemoryLifecyclePersistence({"sub-child-crash": _created_record()})
    provider = _CrashAfterChildPostProvider()
    coordinator = SubscriptionLifecycleCoordinator(persistence, provider)
    command = CreateSubscription(CommandMetadata("create-child-crash"), PLAN_LOCAL)

    with pytest.raises(RuntimeError, match="child POST"):
        await coordinator.apply(
            command,
            "create-child-crash",
            subscription_id="sub-child-crash",
            context=PolicyContext(now=NOW),
        )

    retried = await coordinator.apply(
        command,
        "create-child-crash",
        subscription_id="sub-child-crash",
        context=PolicyContext(now=NOW),
    )

    assert isinstance(retried, Success)
    assert provider.subscription_posts == 1
    child_keys = [key for key in persistence.intents if key.endswith(":subscription")]
    assert len(child_keys) == 1
    assert persistence.intents[child_keys[0]].status is IntentStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_scheduled_pending_recovery_uses_explicit_provider_plan_binding() -> None:
    """The recovery sweep must execute durable PENDING work with provider IDs."""
    persistence = InMemoryLifecyclePersistence({"sub-child-crash": _created_record()})
    operation = CreateSubscriptionOperation(
        plan_id=PLAN_PROVIDER,
        customer_id="cust-1",
        total_count=0,
        quantity=1,
        customer_notify=True,
        correlation_id="scheduled-create",
    )
    persistence.intents["scheduled-create"] = ProviderOperationIntent(
        key="scheduled-create",
        subscription_id="sub-child-crash",
        operation=operation,
    )
    provider = InMemoryBillingProvider()
    recovered = await SubscriptionLifecycleCoordinator(persistence, provider).recover_pending()

    assert recovered == Success(1)
    assert persistence.intents["scheduled-create"].status is IntentStatus.SUCCEEDED
    calls = [call.operation for call in provider.calls]
    create_call = next(
        call for call in calls if call.kind is ProviderOperationKind.CREATE_SUBSCRIPTION
    )
    assert create_call.plan_id == PLAN_PROVIDER


def _cancel_subscription() -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(),
        user_id="user-1",
        plan_id="plan-1",
        status=SubscriptionStatus.CANCELLED.value,
        version=4,
        current_period_start=datetime(2026, 10, 1, tzinfo=UTC),
        current_period_end=datetime(2026, 11, 1, tzinfo=UTC),
        trial_end=None,
        cancel_at_period_end=False,
        pause_start=None,
        pause_end=None,
        retry_count=0,
        currency_display="INR",
        remote_sync="pending",
        observed_plan_id=None,
        entitlement_until=None,
        cancellation_requested_at=None,
        razorpay_subscription_id="sub-remote",
        created_at=NOW,
        updated_at=NOW,
        metadata_={},
    )


def _cancel_service(
    subscription: SimpleNamespace, payments: object, coordinator: object
) -> SubscriptionService:
    plan = SimpleNamespace(
        id="plan-1",
        name="Pro",
        amount=1_000,
        interval="monthly",
        currency="INR",
        refund_policy="FULL",
    )
    subscriptions = SimpleNamespace(
        find_by_id=AsyncMock(side_effect=[Success(subscription), Success(subscription)]),
        update_with_lock=AsyncMock(return_value=Success(subscription)),
    )
    plans = SimpleNamespace(find_by_id=AsyncMock(return_value=Success(plan)))
    audit = SimpleNamespace(create=AsyncMock(return_value=Success(None)))
    return SubscriptionService(
        object(),
        subscriptions,  # type: ignore[arg-type]
        plans,  # type: ignore[arg-type]
        payments,  # type: ignore[arg-type]
        audit,  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        coordinator,  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_cancel_payment_lookup_failure_surfaces_manual_review() -> None:
    subscription = _cancel_subscription()
    payments = SimpleNamespace(
        find_by_subscription=AsyncMock(
            return_value=Failure(
                SubscriptionInfrastructureError(
                    message="payments unavailable", source="payment_repository", operation="find"
                )
            )
        )
    )
    service = _cancel_service(subscription, payments, SimpleNamespace())

    result = await service.cancel_subscription(
        "user-1", str(subscription.id), SubscriptionCancelDTO(cancel_at_period_end=False)
    )

    assert isinstance(result, Failure)
    assert isinstance(result.failure(), SubscriptionRefundError)
    assert result.failure().details["manual_review"] is True


@pytest.mark.asyncio
async def test_cancel_chooses_latest_captured_payment_before_newer_failed_row() -> None:
    subscription = _cancel_subscription()
    captured_old = SimpleNamespace(
        id=uuid4(),
        subscription_id=subscription.id,
        razorpay_payment_id="pay-captured-old",
        amount=1_000,
        status=PaymentStatus.CAPTURED.value,
        refund_amount=Decimal(0),
        metadata_={},
        currency="INR",
        captured_at=datetime(2026, 10, 1, tzinfo=UTC),
        created_at=datetime(2026, 10, 1, 1, tzinfo=UTC),
    )
    captured_current = SimpleNamespace(
        id=uuid4(),
        subscription_id=subscription.id,
        razorpay_payment_id="pay-captured-current",
        amount=1_500,
        status=PaymentStatus.PARTIALLY_REFUNDED.value,
        refund_amount=Decimal("5.00"),
        metadata_={},
        currency="INR",
        captured_at=datetime(2026, 10, 15, tzinfo=UTC),
        created_at=datetime(2026, 10, 15, 1, tzinfo=UTC),
    )
    failed = SimpleNamespace(
        id=uuid4(),
        subscription_id=subscription.id,
        razorpay_payment_id="pay-failed-new",
        amount=0,
        status=PaymentStatus.FAILED.value,
        refund_amount=Decimal(0),
        metadata_={},
        currency="INR",
        captured_at=datetime(2026, 10, 20, tzinfo=UTC),
        created_at=datetime(2026, 10, 20, 1, tzinfo=UTC),
    )

    def refundable_rows(_id: object, *, limit: int | None = None):
        rows = [failed, captured_old, captured_current]
        return Success(rows if limit is None else rows[:limit])

    payments = SimpleNamespace(
        find_by_subscription=AsyncMock(side_effect=refundable_rows),
        find_refundable_captured_by_subscription=AsyncMock(
            return_value=Success([failed, captured_old, captured_current])
        ),
        update_status=AsyncMock(return_value=Success(captured_current)),
    )
    coordinator = SimpleNamespace(
        execute_provider_operation=AsyncMock(
            return_value=Success(
                ProviderSuccess(
                    ProviderOperationKind.CREATE_REFUND,
                    "cancel-refund",
                    ProviderReceipt("rfnd-current-capture", "pay-captured-current", 1_000),
                )
            )
        )
    )
    service = _cancel_service(subscription, payments, coordinator)

    result = await service.cancel_subscription(
        "user-1", str(subscription.id), SubscriptionCancelDTO(cancel_at_period_end=False)
    )

    assert isinstance(result, Success)
    payments.find_refundable_captured_by_subscription.assert_awaited_once_with(subscription.id)
    coordinator.execute_provider_operation.assert_awaited_once()
    assert (
        coordinator.execute_provider_operation.await_args.args[0].payment_id
        == "pay-captured-current"
    )


@pytest.mark.asyncio
async def test_cancel_refund_uses_economic_key_across_distinct_request_keys() -> None:
    """Concurrent retries with new HTTP keys cannot issue a second refund."""
    subscription = _cancel_subscription()
    payment = SimpleNamespace(
        id=uuid4(),
        subscription_id=subscription.id,
        razorpay_payment_id="pay-cancel-once",
        amount=1_000,
        status=PaymentStatus.CAPTURED.value,
        refund_amount=Decimal(0),
        metadata_={},
        currency="INR",
        captured_at=datetime(2026, 10, 15, tzinfo=UTC),
        created_at=datetime(2026, 10, 15, 1, tzinfo=UTC),
    )

    class _GatedRefundProvider(InMemoryBillingProvider):
        def __init__(self) -> None:
            super().__init__()
            self.create_started = asyncio.Event()
            self.release_create = asyncio.Event()
            self.create_done = asyncio.Event()
            self.refund_posts = 0

        async def execute(self, operation: Any):  # type: ignore[no-untyped-def]
            if operation.kind is ProviderOperationKind.CREATE_REFUND:
                self.refund_posts += 1
                self.create_started.set()
                await self.release_create.wait()
                result = await super().execute(operation)
                self.create_done.set()
                return result
            if operation.kind is ProviderOperationKind.FIND_REFUND_BY_CORRELATION:
                # The losing caller must reconcile the winner's receipt; it
                # must not turn the claim race into another POST.
                await self.create_done.wait()
            return await super().execute(operation)

    provider = _GatedRefundProvider()
    persistence = InMemoryLifecyclePersistence(
        {
            str(subscription.id): LifecycleRecord(
                snapshot=SubscriptionSnapshot(
                    id=str(subscription.id),
                    user_id=subscription.user_id,
                    plan_id=str(subscription.plan_id),
                    state=ActiveState(),
                )
            )
        }
    )
    coordinator = SubscriptionLifecycleCoordinator(persistence, provider)

    def update_payment(
        candidate: object, *, status: str, extra_values: dict[str, object] | None = None
    ) -> Success[object]:
        del status
        values = extra_values or {}
        payment.refund_amount = values["refund_amount"]
        payment.metadata_ = values["metadata_"]
        return Success(candidate)

    payments = SimpleNamespace(
        find_refundable_captured_by_subscription=AsyncMock(return_value=Success([payment])),
        update_status=AsyncMock(side_effect=update_payment),
    )
    service = _cancel_service(subscription, payments, coordinator)

    first = asyncio.create_task(
        service._refund_on_cancel(
            subscription,
            SimpleNamespace(refund_policy="FULL"),
            "user-1",
            idempotency_key="request-A",
        )
    )
    await provider.create_started.wait()
    second = asyncio.create_task(
        service._refund_on_cancel(
            subscription,
            SimpleNamespace(refund_policy="FULL"),
            "user-1",
            idempotency_key="request-B",
        )
    )
    await asyncio.sleep(0)
    provider.release_create.set()
    results = await asyncio.gather(first, second)

    assert all(isinstance(result, Success) for result in results)
    assert sorted(result.unwrap() for result in results) == [0, 1_000]
    assert provider.refund_posts == 1
    assert payment.refund_amount == 10
    assert payment.metadata_["processed_refund_ids"] == ["rfnd_memory_1"]
    assert list(persistence.intents) == [f"billing:cancel-refund:{subscription.id}:{payment.id}"]


@pytest.mark.asyncio
async def test_cancel_refund_retry_reconciles_crash_after_remote_receipt() -> None:
    """A crash after POST is recovered by correlation read, not a second POST."""
    subscription = _cancel_subscription()
    payment = SimpleNamespace(
        id=uuid4(),
        subscription_id=subscription.id,
        razorpay_payment_id="pay-cancel-crash",
        amount=1_000,
        status=PaymentStatus.CAPTURED.value,
        refund_amount=Decimal(0),
        metadata_={},
        currency="INR",
        captured_at=datetime(2026, 10, 15, tzinfo=UTC),
        created_at=datetime(2026, 10, 15, 1, tzinfo=UTC),
    )

    class _CrashAfterRefundProvider(InMemoryBillingProvider):
        def __init__(self) -> None:
            super().__init__()
            self.refund_posts = 0
            self.crashed = False

        async def execute(self, operation: Any):  # type: ignore[no-untyped-def]
            if operation.kind is ProviderOperationKind.CREATE_REFUND:
                self.refund_posts += 1
                result = await super().execute(operation)
                if not self.crashed:
                    self.crashed = True
                    message = "worker crashed after refund POST"
                    raise RuntimeError(message)
                return result
            return await super().execute(operation)

    provider = _CrashAfterRefundProvider()
    persistence = InMemoryLifecyclePersistence(
        {
            str(subscription.id): LifecycleRecord(
                snapshot=SubscriptionSnapshot(
                    id=str(subscription.id),
                    user_id=subscription.user_id,
                    plan_id=str(subscription.plan_id),
                    state=ActiveState(),
                )
            )
        }
    )
    coordinator = SubscriptionLifecycleCoordinator(persistence, provider)

    def update_payment(
        candidate: object, *, status: str, extra_values: dict[str, object] | None = None
    ) -> Success[object]:
        del status
        values = extra_values or {}
        payment.refund_amount = values["refund_amount"]
        payment.metadata_ = values["metadata_"]
        return Success(candidate)

    payments = SimpleNamespace(
        find_refundable_captured_by_subscription=AsyncMock(return_value=Success([payment])),
        update_status=AsyncMock(side_effect=update_payment),
    )
    service = _cancel_service(subscription, payments, coordinator)

    first = await service._refund_on_cancel(
        subscription,
        SimpleNamespace(refund_policy="FULL"),
        "user-1",
        idempotency_key="request-A",
    )
    second = await service._refund_on_cancel(
        subscription,
        SimpleNamespace(refund_policy="FULL"),
        "user-1",
        idempotency_key="request-B",
    )

    assert isinstance(first, Failure)
    assert isinstance(second, Success)
    assert second.unwrap() == 1_000
    assert first.failure().details["refund_status"] == "unknown"
    assert provider.refund_posts == 1
    assert payment.refund_amount == 10
    assert payment.metadata_["processed_refund_ids"] == ["rfnd_memory_1"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("receipt_payment_id", "receipt_amount"),
    [("pay-not-selected", 1_000), ("pay-captured-current", 999)],
)
async def test_cancel_receipt_mismatch_requires_manual_review_without_projection(
    receipt_payment_id: str, receipt_amount: int
) -> None:
    subscription = _cancel_subscription()
    payment = SimpleNamespace(
        id=uuid4(),
        subscription_id=subscription.id,
        razorpay_payment_id="pay-captured-current",
        amount=1_000,
        status=PaymentStatus.CAPTURED.value,
        refund_amount=Decimal(0),
        metadata_={},
        currency="INR",
        captured_at=datetime(2026, 10, 15, tzinfo=UTC),
        created_at=datetime(2026, 10, 15, 1, tzinfo=UTC),
    )
    payments = SimpleNamespace(
        find_refundable_captured_by_subscription=AsyncMock(return_value=Success([payment])),
        update_status=AsyncMock(return_value=Success(payment)),
    )
    coordinator = SimpleNamespace(
        execute_provider_operation=AsyncMock(
            return_value=Success(
                ProviderSuccess(
                    ProviderOperationKind.CREATE_REFUND,
                    "cancel-refund-mismatch",
                    ProviderReceipt("rfnd-mismatch", receipt_payment_id, receipt_amount),
                )
            )
        )
    )
    service = _cancel_service(subscription, payments, coordinator)

    result = await service.cancel_subscription(
        "user-1", str(subscription.id), SubscriptionCancelDTO(cancel_at_period_end=False)
    )

    assert isinstance(result, Failure)
    assert isinstance(result.failure(), SubscriptionRefundError)
    assert result.failure().details["manual_review"] is True
    assert result.failure().details["requires_reconciliation"] is True
    assert payments.update_status.await_count == 0
    assert service.audit.create.await_count == 0  # type: ignore[attr-defined]
    assert subscription.remote_sync == "pending"
    assert payment.refund_amount == Decimal(0)
    assert payment.metadata_ == {}


@pytest.mark.asyncio
async def test_source_invoice_search_is_not_limited_to_the_newest_hundred_payments() -> None:
    subscription = _cancel_subscription()
    source_invoice = uuid4()
    rows = [SimpleNamespace(status=PaymentStatus.FAILED.value, invoice_id=None) for _ in range(100)]
    rows.append(SimpleNamespace(status=PaymentStatus.CAPTURED.value, invoice_id=source_invoice))
    payments = SimpleNamespace(
        find_by_subscription=AsyncMock(
            side_effect=lambda _id, *, limit=None: Success(rows if limit is None else rows[:limit])
        )
    )
    service = _cancel_service(subscription, payments, SimpleNamespace())

    resolved = await service._authoritative_source_invoice_id(subscription)

    assert resolved == Success(source_invoice)
    assert payments.find_by_subscription.await_args.kwargs["limit"] is None


@pytest.mark.asyncio
async def test_source_invoice_query_sorts_undated_captures_last() -> None:
    source_invoice = uuid4()
    session = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(scalar_one_or_none=lambda: source_invoice)),
        rollback=AsyncMock(),
    )
    repository = PaymentRepository(session)

    resolved = await repository.find_oldest_captured_invoice_by_subscription(uuid4())

    assert resolved == Success(source_invoice)
    statement = session.execute.await_args.args[0]
    sql = str(statement.compile(dialect=postgresql.dialect()))
    assert "payments.captured_at ASC NULLS LAST" in sql


@pytest.mark.asyncio
async def test_unknown_refund_replay_keeps_one_remote_mutation_and_one_receipt() -> None:
    class _RefundProvider(InMemoryBillingProvider):
        def __init__(self) -> None:
            super().__init__()
            self.refund_posts = 0
            self.refund_reads = 0

        async def execute(self, operation: Any):  # type: ignore[no-untyped-def]
            if operation.kind is ProviderOperationKind.CREATE_REFUND:
                self.refund_posts += 1
                if self.refund_posts == 1:
                    from app.features.billing.providers import ProviderOutcomeUnknownError

                    return Failure(
                        ProviderOutcomeUnknownError(operation.kind, operation.correlation_id)
                    )
            if operation.kind is ProviderOperationKind.FIND_REFUND_BY_CORRELATION:
                self.refund_reads += 1
                return Success(
                    ProviderSuccess(
                        operation.kind,
                        operation.correlation_id,
                        ProviderReceipt("rfnd-replay", "pay-1", 500),
                    )
                )
            return await super().execute(operation)

    provider = _RefundProvider()
    coordinator = SubscriptionLifecycleCoordinator(InMemoryLifecyclePersistence(), provider)
    operation = CreateRefundOperation("pay-1", 500, "refund-replay")

    first = await coordinator.execute_provider_operation(
        operation, subscription_id=None, idempotency_key="refund-replay"
    )
    second = await coordinator.execute_provider_operation(
        operation, subscription_id=None, idempotency_key="refund-replay"
    )

    assert isinstance(first, Success)
    assert isinstance(second, Success)
    assert first.unwrap().value == second.unwrap().value
    assert provider.refund_posts == 1
    assert provider.refund_reads == 1


@pytest.mark.asyncio
async def test_capture_before_link_reconciliation_is_recovered_before_dispatch() -> None:
    subscription = SimpleNamespace(
        id=uuid4(),
        metadata_={
            "pending_upgrade": {
                "payment_link_key": "link-intent-1",
                "amount_paisa": 1_000,
                "currency": "INR",
            }
        },
    )

    class _RecoveryCoordinator:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def recover_pending_for_capture(
            self,
            *,
            subscription_id: str,
            payment_link_key: str,
            payment_id: str,
            amount_paisa: int,
            currency: str,
        ):
            self.calls.append(
                {
                    "subscription_id": subscription_id,
                    "payment_link_key": payment_link_key,
                    "payment_id": payment_id,
                    "amount_paisa": amount_paisa,
                    "currency": currency,
                }
            )
            return Success(None)

    coordinator = _RecoveryCoordinator()
    service = WebhookService(
        webhooks=SimpleNamespace(),  # type: ignore[arg-type]
        subscriptions=SimpleNamespace(),  # type: ignore[arg-type]
        plans=SimpleNamespace(),  # type: ignore[arg-type]
        audit=SimpleNamespace(),  # type: ignore[arg-type]
        payment_service=SimpleNamespace(),  # type: ignore[arg-type]
        invoice_service=SimpleNamespace(),  # type: ignore[arg-type]
        lifecycle_coordinator=coordinator,  # type: ignore[arg-type]
    )

    recovered = await service._recover_pending_upgrade_for_capture(  # type: ignore[attr-defined]
        subscription,
        payment_id="pay-before-link",
        amount_paisa=1_000,
        currency="INR",
    )

    assert recovered == Success("link-intent-1")
    assert coordinator.calls == [
        {
            "subscription_id": str(subscription.id),
            "payment_link_key": "link-intent-1",
            "payment_id": "pay-before-link",
            "amount_paisa": 1_000,
            "currency": "INR",
        }
    ]


class _CaptureRecoveryProvider(InMemoryBillingProvider):
    def __init__(self, *, fail_target: bool = False) -> None:
        super().__init__()
        self.fail_target = fail_target
        self.attempted: list[str] = []

    async def execute(self, operation: Any):  # type: ignore[no-untyped-def]
        if isinstance(operation, CreatePaymentLinkOperation):
            self.attempted.append(operation.correlation_id)
        if (
            isinstance(operation, CreatePaymentLinkOperation)
            and operation.correlation_id == "target-link"
            and self.fail_target
        ):
            return Failure(ProviderUnavailableError())
        if (
            isinstance(operation, CreatePaymentLinkOperation)
            and operation.correlation_id == "unrelated-link"
        ):
            return Failure(ProviderUnavailableError())
        return await super().execute(operation)


def _capture_recovery_intent(
    key: str, *, subscription_id: str = "sub-capture"
) -> ProviderOperationIntent:
    return ProviderOperationIntent(
        key=key,
        subscription_id=subscription_id,
        operation=CreatePaymentLinkOperation(
            amount=1_000,
            currency="INR",
            description="upgrade",
            correlation_id=key,
            notes={
                "upgrade_dependency": "true",
                "amount_paisa": "1000",
                "currency": "INR",
            },
        ),
    )


@pytest.mark.asyncio
async def test_capture_recovery_is_key_scoped_and_ignores_unrelated_failure() -> None:
    persistence = InMemoryLifecyclePersistence(
        {
            "sub-capture": LifecycleRecord(
                SubscriptionSnapshot(
                    "sub-capture", "user-1", "plan-1", ActiveState(NOW, NOW), version=1
                ),
                provider_subscription_id="provider-sub-1",
            )
        }
    )
    persistence.intents["target-link"] = _capture_recovery_intent("target-link")
    persistence.intents["unrelated-link"] = _capture_recovery_intent("unrelated-link")
    provider = _CaptureRecoveryProvider()
    coordinator = SubscriptionLifecycleCoordinator(persistence, provider)

    result = await coordinator.recover_pending_for_capture(
        subscription_id="sub-capture",
        payment_link_key="target-link",
        payment_id="payment-capture",
        amount_paisa=1_000,
        currency="INR",
    )

    assert isinstance(result, Success)
    assert provider.attempted == ["target-link"]
    assert persistence.intents["target-link"].status is IntentStatus.SUCCEEDED
    assert persistence.intents["unrelated-link"].status is IntentStatus.PENDING


@pytest.mark.asyncio
async def test_webhook_capture_recovery_uses_coordinator_generated_payment_link_key() -> None:
    persistence = InMemoryLifecyclePersistence(
        {
            "sub-capture": LifecycleRecord(
                SubscriptionSnapshot(
                    "sub-capture",
                    "user-1",
                    "plan-1",
                    ActiveState(
                        datetime(2026, 10, 1, tzinfo=UTC),
                        datetime(2026, 11, 1, tzinfo=UTC),
                    ),
                    version=1,
                    observed_plan_id="plan-1",
                ),
                provider_subscription_id="sub_memory_1",
                provider_plan_id="plan-1",
            )
        }
    )
    provider = _CaptureRecoveryProvider()
    await provider.execute(CreateSubscriptionOperation("plan-1", "cust-1", 0, 1, True, "seed"))
    coordinator = SubscriptionLifecycleCoordinator(persistence, provider)
    staged = await coordinator.apply(
        LifecycleRequest(
            "sub-capture",
            ChangeSubscriptionPlan(
                CommandMetadata("capture-command"),
                "plan-2",
                NOW,
            ),
            PolicyContext(
                now=NOW,
                current_plan_amount=1_000,
                target_plan_amount=2_000,
                target_currency="INR",
                target_provider_plan_id="provider-plan-2",
            ),
        ),
        "capture-command",
    )
    assert isinstance(staged, Success)
    pending = persistence.records["sub-capture"].snapshot.pending_upgrade
    assert pending is not None
    payment_link_intents = [
        intent
        for intent in persistence.intents.values()
        if isinstance(intent.operation, CreatePaymentLinkOperation)
    ]
    assert len(payment_link_intents) == 1
    target = payment_link_intents[0]
    assert pending.payment_link_key == target.key

    # Model a capture arriving after the provider POST but before the durable
    # success state was finalized.  The existing lease forces reconciliation
    # to use a provider read, while the unrelated failing intent must remain
    # untouched.
    persistence.intents[target.key] = replace(target, status=IntentStatus.PENDING)
    unrelated = _capture_recovery_intent("unrelated-link", subscription_id="sub-capture")
    persistence.intents[unrelated.key] = unrelated
    subscription = SimpleNamespace(
        id="sub-capture",
        metadata_={
            "pending_upgrade": {
                "payment_link_key": pending.payment_link_key,
                "amount_paisa": pending.amount_paisa,
                "currency": pending.currency,
            }
        },
    )
    service = WebhookService(
        webhooks=SimpleNamespace(),  # type: ignore[arg-type]
        subscriptions=SimpleNamespace(),  # type: ignore[arg-type]
        plans=SimpleNamespace(),  # type: ignore[arg-type]
        audit=SimpleNamespace(),  # type: ignore[arg-type]
        payment_service=SimpleNamespace(),  # type: ignore[arg-type]
        invoice_service=SimpleNamespace(),  # type: ignore[arg-type]
        lifecycle_coordinator=coordinator,
    )

    recovered = await service._recover_pending_upgrade_for_capture(  # type: ignore[attr-defined]
        subscription,
        payment_id="payment-capture",
        amount_paisa=pending.amount_paisa,
        currency=pending.currency,
    )

    assert recovered == Success(pending.payment_link_key)
    assert persistence.intents[target.key].status is IntentStatus.SUCCEEDED
    assert persistence.intents[unrelated.key].status is IntentStatus.PENDING
    assert all(
        getattr(call.operation, "correlation_id", None) != "unrelated-link"
        for call in provider.calls
    )


@pytest.mark.asyncio
async def test_capture_completion_pairs_exact_production_operation_key() -> None:
    """A same-priced sibling link must never become the captured upgrade."""
    subscription_id = uuid4()
    target_key = "target-payment-link"
    target = ProviderOperationRecord(
        id=uuid4(),
        subscription_id=subscription_id,
        correlation_id=target_key,
        idempotency_key=target_key,
        operation_kind="create_payment_link",
        state=ProviderOperationState.SUCCEEDED.value,
        observation_data={},
        request_data={
            "operation": {
                "kind": "create_payment_link",
                "fields": {
                    "amount": 1_000,
                    "currency": "INR",
                    "notes": {
                        "subscription_id": str(subscription_id),
                        "upgrade_dependency": "true",
                        "target_plan_id": "target-plan",
                        "target_provider_plan_id": "rp_target-plan",
                        "amount_paisa": "1000",
                        "currency": "INR",
                    },
                },
            }
        },
    )
    unrelated = ProviderOperationRecord(
        id=uuid4(),
        subscription_id=subscription_id,
        correlation_id="unrelated-payment-link",
        idempotency_key="unrelated-payment-link",
        operation_kind="create_payment_link",
        state=ProviderOperationState.SUCCEEDED.value,
        observation_data={},
        request_data={
            "operation": {
                "kind": "create_payment_link",
                "fields": {
                    "amount": 1_000,
                    "currency": "INR",
                    "notes": {
                        "subscription_id": str(subscription_id),
                        "upgrade_dependency": "true",
                        "target_plan_id": "unrelated-plan",
                        "target_provider_plan_id": "rp_unrelated-plan",
                        "amount_paisa": "1000",
                        "currency": "INR",
                    },
                },
            }
        },
    )

    class _Session:
        def __init__(self) -> None:
            self.statements: list[object] = []
            self.rows = {target.idempotency_key: target, unrelated.idempotency_key: unrelated}

        async def execute(self, statement: object):
            self.statements.append(statement)
            if len(self.statements) == 1:
                params = cast("Any", statement).compile().params
                requested_key = next(value for value in params.values() if value in self.rows)
                return SimpleNamespace(scalar_one_or_none=lambda: self.rows[requested_key])
            return SimpleNamespace(scalar_one_or_none=lambda: target)

        async def commit(self) -> None:
            return None

    class _Coordinator:
        def __init__(self) -> None:
            self.dependencies: list[CapturedUpgrade] = []
            self.captured_payment: str | None = None

        async def complete_captured_upgrade(self, dependency: CapturedUpgrade):
            if self.captured_payment is None:
                self.captured_payment = dependency.payment_id
            elif self.captured_payment != dependency.payment_id:
                return Failure(
                    LifecycleCoordinatorTransientProviderError(
                        message="capture pairing conflict",
                        source="test",
                        operation=ProviderOperationKind.CHANGE_SUBSCRIPTION_PLAN,
                    )
                )
            self.dependencies.append(dependency)
            return Success(None)

    session = _Session()
    coordinator = _Coordinator()
    service = WebhookService(
        webhooks=SimpleNamespace(session=session),  # type: ignore[arg-type]
        subscriptions=SimpleNamespace(),  # type: ignore[arg-type]
        plans=SimpleNamespace(),  # type: ignore[arg-type]
        audit=SimpleNamespace(),  # type: ignore[arg-type]
        payment_service=SimpleNamespace(),  # type: ignore[arg-type]
        invoice_service=SimpleNamespace(),  # type: ignore[arg-type]
        lifecycle_coordinator=coordinator,  # type: ignore[arg-type]
    )
    subscription = SimpleNamespace(
        id=subscription_id,
        metadata_={
            "pending_upgrade": {
                "payment_link_key": target_key,
                "target_plan_id": "target-plan",
                "target_provider_plan_id": "rp_target-plan",
            }
        },
    )

    result = await service._complete_captured_upgrade(  # type: ignore[attr-defined]
        subscription,
        payment_id="captured-payment",
        amount_paisa=1_000,
        currency="INR",
        payment_link_key=target_key,
    )

    assert result == Success(None)
    assert coordinator.dependencies[0].payment_link_key == target_key
    assert coordinator.dependencies[0].target_plan_id == "target-plan"
    assert unrelated.observation_data == {}
    assert "ORDER BY" not in str(session.statements[0]).upper()


@pytest.mark.asyncio
async def test_capture_pairing_cas_rejects_concurrent_distinct_payment_ids() -> None:
    """Only one capture may claim a payment-link operation under concurrency."""
    subscription_id = uuid4()
    payment_link_key = "capture-pair-cas"
    row = SimpleNamespace(
        id=uuid4(),
        subscription_id=subscription_id,
        operation_kind="create_payment_link",
        state=ProviderOperationState.SUCCEEDED.value,
        observation_data={},
        request_data={
            "operation": {
                "kind": "create_payment_link",
                "fields": {
                    "amount": 1_000,
                    "currency": "INR",
                    "notes": {
                        "subscription_id": str(subscription_id),
                        "upgrade_dependency": "true",
                        "target_plan_id": "target-plan",
                        "target_provider_plan_id": "rp_target-plan",
                        "amount_paisa": "1000",
                        "currency": "INR",
                    },
                },
            }
        },
    )

    class _PairStore:
        def __init__(self) -> None:
            self.lock = asyncio.Lock()
            self.reads = 0
            self.reads_ready = asyncio.Event()
            self.captured_payment: str | None = None

    class _Session:
        def __init__(self, store: _PairStore, payment_id: str) -> None:
            self.store = store
            self.payment_id = payment_id
            self.calls = 0

        async def execute(self, statement: object):
            del statement
            self.calls += 1
            if self.calls == 1:
                self.store.reads += 1
                if self.store.reads == 2:
                    self.store.reads_ready.set()
                await self.store.reads_ready.wait()
                return SimpleNamespace(
                    scalar_one_or_none=lambda: SimpleNamespace(
                        **{**vars(row), "observation_data": {}}
                    )
                )
            if self.calls == 2:
                async with self.store.lock:
                    if self.store.captured_payment is not None:
                        return SimpleNamespace(scalar_one_or_none=lambda: None)
                    self.store.captured_payment = self.payment_id
                    row.observation_data = {"captured_payment_id": self.payment_id}
                    return SimpleNamespace(scalar_one_or_none=lambda: row)
            return SimpleNamespace(scalar_one_or_none=lambda: row)

        async def commit(self) -> None:
            return None

        async def rollback(self) -> None:
            return None

    class _Coordinator:
        def __init__(self) -> None:
            self.dependencies: list[CapturedUpgrade] = []
            self.captured_payment: str | None = None

        async def complete_captured_upgrade(self, dependency: CapturedUpgrade):
            if self.captured_payment is None:
                self.captured_payment = dependency.payment_id
            elif self.captured_payment != dependency.payment_id:
                return Failure(
                    LifecycleCoordinatorTransientProviderError(
                        message="capture pairing conflict",
                        source="test",
                        operation=ProviderOperationKind.CHANGE_SUBSCRIPTION_PLAN,
                    )
                )
            self.dependencies.append(dependency)
            return Success(None)

    store = _PairStore()
    coordinator = _Coordinator()
    subscription = SimpleNamespace(
        id=subscription_id,
        metadata_={
            "pending_upgrade": {
                "payment_link_key": payment_link_key,
                "target_plan_id": "target-plan",
                "target_provider_plan_id": "rp_target-plan",
            }
        },
    )

    async def complete(payment_id: str):
        service = WebhookService(
            webhooks=SimpleNamespace(session=_Session(store, payment_id)),
            subscriptions=SimpleNamespace(),  # type: ignore[arg-type]
            plans=SimpleNamespace(),  # type: ignore[arg-type]
            audit=SimpleNamespace(),  # type: ignore[arg-type]
            payment_service=SimpleNamespace(),  # type: ignore[arg-type]
            invoice_service=SimpleNamespace(),  # type: ignore[arg-type]
            lifecycle_coordinator=coordinator,  # type: ignore[arg-type]
        )
        return await service._complete_captured_upgrade(  # type: ignore[attr-defined]
            subscription,
            payment_id=payment_id,
            amount_paisa=1_000,
            currency="INR",
            payment_link_key=payment_link_key,
        )

    results = await asyncio.gather(complete("capture-A"), complete("capture-B"))

    assert sum(isinstance(result, Success) for result in results) == 1
    assert sum(isinstance(result, Failure) for result in results) == 1
    assert coordinator.captured_payment in {"capture-A", "capture-B"}
    assert len(coordinator.dependencies) == 1


async def _capture_completion_fixture(
    provider: InMemoryBillingProvider,
) -> tuple[SubscriptionLifecycleCoordinator, InMemoryLifecyclePersistence]:
    persistence = InMemoryLifecyclePersistence(
        {
            "sub-capture-pair": LifecycleRecord(
                snapshot=SubscriptionSnapshot(
                    "sub-capture-pair", "user-1", PLAN_LOCAL, ActiveState(), version=1
                ),
                provider_subscription_id="sub_memory_1",
                provider_plan_id="rp_current",
                desired_provider_plan_id="rp_current",
            )
        }
    )
    link = CreatePaymentLinkOperation(
        amount=1_000,
        currency="INR",
        description="upgrade",
        correlation_id="capture-pair-link",
        notes={"subscription_id": "sub-capture-pair"},
    )
    persistence.intents["capture-pair-link"] = ProviderOperationIntent(
        key="capture-pair-link",
        subscription_id="sub-capture-pair",
        operation=link,
        status=IntentStatus.SUCCEEDED,
        result=ProviderSuccess(
            ProviderOperationKind.CREATE_PAYMENT_LINK,
            link.correlation_id,
            ProviderPaymentLink("plink-capture-pair", "https://example.test/pay/capture-pair"),
        ),
    )
    await provider.execute(
        CreateSubscriptionOperation("rp_current", "cust-capture-pair", 0, 1, True, "seed-pair")
    )
    return SubscriptionLifecycleCoordinator(persistence, provider), persistence


@pytest.mark.asyncio
async def test_coordinator_capture_pair_claim_allows_one_distinct_payment_id() -> None:
    class _GatedProvider(InMemoryBillingProvider):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.plan_changes = 0

        async def execute(self, operation: Any):  # type: ignore[no-untyped-def]
            if operation.kind is ProviderOperationKind.CHANGE_SUBSCRIPTION_PLAN:
                self.plan_changes += 1
                self.started.set()
                await self.release.wait()
            return await super().execute(operation)

    provider = _GatedProvider()
    coordinator, persistence = await _capture_completion_fixture(provider)

    async def complete(payment_id: str):
        return await coordinator.complete_captured_upgrade(
            CapturedUpgrade(
                subscription_id="sub-capture-pair",
                payment_link_key="capture-pair-link",
                payment_id=payment_id,
                target_plan_id="plan-target",
                target_provider_plan_id="rp_target",
                amount_paisa=1_000,
                currency="INR",
            )
        )

    first = asyncio.create_task(complete("capture-A"))
    await provider.started.wait()
    second = asyncio.create_task(complete("capture-B"))
    await asyncio.sleep(0)
    provider.release.set()
    results = await asyncio.gather(first, second)

    assert sum(isinstance(result, Success) for result in results) == 1
    assert sum(isinstance(result, Failure) for result in results) == 1
    assert provider.plan_changes == 1
    assert persistence.capture_pairs == {"capture-pair-link": "capture-A"}


@pytest.mark.asyncio
async def test_coordinator_capture_pair_survives_crash_before_remote_outcome() -> None:
    class _CrashAfterPlanProvider(InMemoryBillingProvider):
        def __init__(self) -> None:
            super().__init__()
            self.plan_changes = 0
            self.crashed = False

        async def execute(self, operation: Any):  # type: ignore[no-untyped-def]
            if operation.kind is ProviderOperationKind.CHANGE_SUBSCRIPTION_PLAN:
                self.plan_changes += 1
                result = await super().execute(operation)
                if not self.crashed:
                    self.crashed = True
                    message = "worker crashed after capture pair before ledger outcome"
                    raise RuntimeError(message)
                return result
            return await super().execute(operation)

    provider = _CrashAfterPlanProvider()
    coordinator, persistence = await _capture_completion_fixture(provider)
    dependency = CapturedUpgrade(
        subscription_id="sub-capture-pair",
        payment_link_key="capture-pair-link",
        payment_id="capture-A",
        target_plan_id="plan-target",
        target_provider_plan_id="rp_target",
        amount_paisa=1_000,
        currency="INR",
    )

    with pytest.raises(RuntimeError, match="capture pair"):
        await coordinator.complete_captured_upgrade(dependency)
    recovered = await coordinator.complete_captured_upgrade(dependency)

    assert isinstance(recovered, Success)
    assert provider.plan_changes == 1
    assert persistence.capture_pairs == {"capture-pair-link": "capture-A"}


@pytest.mark.asyncio
async def test_in_memory_capture_claim_replay_returns_false_without_reclaiming() -> None:
    coordinator, persistence = await _capture_completion_fixture(InMemoryBillingProvider())
    del coordinator

    first = await persistence.claim_capture_payment(
        payment_link_key="capture-pair-link",
        subscription_id="sub-capture-pair",
        payment_id="capture-A",
    )
    replay = await persistence.claim_capture_payment(
        payment_link_key="capture-pair-link",
        subscription_id="sub-capture-pair",
        payment_id="capture-A",
    )
    conflict = await persistence.claim_capture_payment(
        payment_link_key="capture-pair-link",
        subscription_id="sub-capture-pair",
        payment_id="capture-B",
    )
    invalid = await InMemoryLifecyclePersistence().claim_capture_payment(
        payment_link_key="missing-link",
        subscription_id="sub-capture-pair",
        payment_id="capture-A",
    )

    assert first == Success(True)
    assert replay == Success(False)
    assert isinstance(conflict, Failure)
    assert conflict.failure().details["manual_review"] is True
    assert isinstance(invalid, Failure)
    assert invalid.failure().details["manual_review"] is True


@pytest.mark.asyncio
async def test_sql_capture_claim_uses_null_only_cas_and_same_id_replay() -> None:
    subscription_id = str(uuid4())

    class _Result:
        def __init__(self, *, rowcount: int, row: object | None = None) -> None:
            self.rowcount = rowcount
            self.row = row

        def scalar_one_or_none(self) -> object | None:
            return self.row

    class _Session:
        def __init__(self, *, rowcount: int, row: object | None = None) -> None:
            self.rowcount = rowcount
            self.row = row
            self.statements: list[object] = []

        async def execute(self, statement: object) -> _Result:
            self.statements.append(statement)
            if len(self.statements) == 1:
                return _Result(rowcount=self.rowcount)
            return _Result(rowcount=0, row=self.row)

        async def commit(self) -> None:
            return None

        async def rollback(self) -> None:
            return None

    first_session = _Session(rowcount=1)
    first = await SqlAlchemyLifecyclePersistence(first_session).claim_capture_payment(
        payment_link_key="capture-sql-link",
        subscription_id=subscription_id,
        payment_id="capture-A",
    )
    sql = str(first_session.statements[0].compile(dialect=postgresql.dialect())).upper()

    replay_row = SimpleNamespace(
        subscription_id=subscription_id,
        operation_kind="create_payment_link",
        state=ProviderOperationState.SUCCEEDED.value,
        observation_data={"captured_payment_id": "capture-A"},
    )
    replay_session = _Session(rowcount=0, row=replay_row)
    replay = await SqlAlchemyLifecyclePersistence(replay_session).claim_capture_payment(
        payment_link_key="capture-sql-link",
        subscription_id=subscription_id,
        payment_id="capture-A",
    )

    assert first == Success(True)
    assert replay == Success(False)
    assert "IS NULL" in sql
    assert " OR " not in sql
    assert len(replay_session.statements) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("row", "payment_id"),
    [
        (
            SimpleNamespace(
                subscription_id="sub-sql",
                operation_kind="create_payment_link",
                state=ProviderOperationState.SUCCEEDED.value,
                observation_data={"captured_payment_id": "capture-A"},
            ),
            "capture-B",
        ),
        (
            SimpleNamespace(
                subscription_id="sub-sql",
                operation_kind="change_subscription_plan",
                state=ProviderOperationState.SUCCEEDED.value,
                observation_data={},
            ),
            "capture-A",
        ),
    ],
)
async def test_sql_capture_claim_rejects_conflict_or_invalid_link_as_manual_review(
    row: object, payment_id: str
) -> None:
    class _Result:
        rowcount = 0

        def scalar_one_or_none(self) -> object:
            return row

    class _Session:
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, statement: object) -> _Result:
            del statement
            self.calls += 1
            return _Result()

        async def commit(self) -> None:
            return None

        async def rollback(self) -> None:
            return None

    session = _Session()
    result = await SqlAlchemyLifecyclePersistence(session).claim_capture_payment(
        payment_link_key="capture-sql-link",
        subscription_id="sub-sql",
        payment_id=payment_id,
    )

    assert isinstance(result, Failure)
    assert result.failure().details["manual_review"] is True
    assert session.calls == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payment_id", "amount", "currency"),
    [("", 1_000, "INR"), ("pay-malformed", 0, "INR"), ("pay-malformed", 1_000, None)],
)
async def test_webhook_malformed_capture_fails_before_lifecycle_staging(
    payment_id: str, amount: object, currency: object
) -> None:
    subscription = SimpleNamespace(id=uuid4(), metadata_={})

    class _Coordinator:
        def __init__(self) -> None:
            self.calls = 0

        async def observe(self, *args: object, **kwargs: object):
            del args, kwargs
            self.calls += 1
            return Success(None)

    coordinator = _Coordinator()
    service = WebhookService(
        webhooks=SimpleNamespace(),  # type: ignore[arg-type]
        subscriptions=SimpleNamespace(
            find_by_razorpay_id=AsyncMock(return_value=Success(subscription))
        ),
        plans=SimpleNamespace(),  # type: ignore[arg-type]
        audit=SimpleNamespace(),  # type: ignore[arg-type]
        payment_service=SimpleNamespace(),  # type: ignore[arg-type]
        invoice_service=SimpleNamespace(),  # type: ignore[arg-type]
        lifecycle_coordinator=coordinator,  # type: ignore[arg-type]
    )

    result = await service._handle_payment_captured(  # type: ignore[attr-defined]
        {
            "payment": {
                "entity": {
                    "id": payment_id,
                    "amount": amount,
                    "currency": currency,
                    "subscription_id": "sub-provider",
                }
            }
        }
    )

    assert isinstance(result, Failure)
    assert coordinator.calls == 0


@pytest.mark.asyncio
async def test_reconciliation_retries_existing_payment_while_upgrade_is_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.features.billing import jobs

    subscription_id = uuid4()
    subscription = SimpleNamespace(
        id=subscription_id,
        metadata_={"pending_upgrade": {"payment_link_key": "pending-link"}},
        version=3,
    )
    local_payment = SimpleNamespace(id=uuid4(), subscription_id=subscription_id)
    provider = InMemoryBillingProvider(
        payments=(
            ProviderPaymentState(
                provider_payment_id="pay-existing",
                status="captured",
                amount=1_000,
                currency="INR",
                subscription_id="sub-provider",
                captured_at=NOW,
            ),
        )
    )

    class _Payments:
        def __init__(self) -> None:
            self.lookups = 0

        async def find_by_razorpay_id(self, _payment_id: str):
            self.lookups += 1
            return Success(local_payment)

    payments = _Payments()

    class _PaymentService:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            self.payments = payments

    class _Subscriptions:
        async def find_by_id(self, _subscription_id: object):
            return Success(subscription)

        async def find_by_razorpay_id(self, _subscription_id: str):
            return Success(subscription)

    class _Lifecycle:
        attempts = 0

        async def observe(self, *args: object, **kwargs: object):
            del args, kwargs
            self.attempts += 1
            if self.attempts == 1:
                return Failure(
                    LifecycleCoordinatorTransientProviderError(
                        message="pending payment-link recovery is retryable",
                        source="test",
                        operation=ProviderOperationKind.CREATE_PAYMENT_LINK,
                    )
                )
            return Success(SimpleNamespace())

    lifecycle = _Lifecycle()

    @asynccontextmanager
    async def _independent_session(_factory: object):
        yield SimpleNamespace()

    monkeypatch.setattr(jobs, "PaymentService", _PaymentService)
    monkeypatch.setattr(
        jobs, "SubscriptionLifecycleCoordinator", lambda _persistence, _provider: lifecycle
    )
    monkeypatch.setattr(jobs, "_subscription_repo", lambda _session: _Subscriptions())
    monkeypatch.setattr(
        jobs,
        "_audit_repo",
        lambda _session: SimpleNamespace(create=AsyncMock(return_value=Success(None))),
    )
    monkeypatch.setattr(jobs, "independent_session", _independent_session)
    monkeypatch.setattr(
        jobs,
        "get_settings",
        lambda: SimpleNamespace(BILLING_RECONCILIATION_LOOKBACK_DAYS=1),
    )

    first = await jobs.reconciliation_job(object(), object(), provider=provider)
    second = await jobs.reconciliation_job(object(), object(), provider=provider)

    assert first["missing"] == 1
    assert second["reconciled"] == 1
    assert lifecycle.attempts == 2
    assert payments.lookups == 2


@pytest.mark.asyncio
async def test_capture_recovery_preserves_retryable_target_failure() -> None:
    persistence = InMemoryLifecyclePersistence(
        {
            "sub-capture": LifecycleRecord(
                SubscriptionSnapshot(
                    "sub-capture", "user-1", "plan-1", ActiveState(NOW, NOW), version=1
                )
            )
        }
    )
    persistence.intents["target-link"] = _capture_recovery_intent("target-link")
    persistence.intents["unrelated-link"] = _capture_recovery_intent("unrelated-link")
    provider = _CaptureRecoveryProvider(fail_target=True)
    coordinator = SubscriptionLifecycleCoordinator(persistence, provider)

    result = await coordinator.recover_pending_for_capture(
        subscription_id="sub-capture",
        payment_link_key="target-link",
        payment_id="payment-capture",
        amount_paisa=1_000,
        currency="INR",
    )

    assert isinstance(result, Failure)
    assert isinstance(result.failure(), LifecycleCoordinatorTransientProviderError)
    assert result.failure().retryable is True
    assert provider.attempted == ["target-link"]
    assert persistence.intents["unrelated-link"].status is IntentStatus.PENDING


@pytest.mark.asyncio
async def test_post_capture_credit_settlement_refunds_once_and_preserves_invoice_cash_math() -> (
    None
):
    payment = SimpleNamespace(
        id=uuid4(),
        subscription_id="sub-settlement",
        razorpay_payment_id="pay-settlement",
        amount=1_000,
        currency="INR",
        status=PaymentStatus.CAPTURED.value,
        refund_amount=Decimal(0),
        metadata_={},
    )

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
            "sub-settlement": LifecycleRecord(
                snapshot=SubscriptionSnapshot(
                    id="sub-settlement",
                    user_id="user-1",
                    plan_id=PLAN_LOCAL,
                    state=ActiveState(
                        datetime(2026, 10, 1, tzinfo=UTC),
                        datetime(2026, 11, 1, tzinfo=UTC),
                    ),
                    version=1,
                ),
                provider_subscription_id="sub-remote",
                provider_plan_id=PLAN_PROVIDER,
            )
        }
    )
    provider = InMemoryBillingProvider()
    coordinator = SubscriptionLifecycleCoordinator(persistence, provider)
    service = PaymentService(
        payments=_Payments(),  # type: ignore[arg-type]
        audit=SimpleNamespace(create=AsyncMock(return_value=Success(None))),  # type: ignore[arg-type]
        lifecycle_coordinator=coordinator,  # type: ignore[arg-type]
    )

    first = await service.settle_credit_application(
        payment,
        invoice_id="invoice-settlement",
        credit_application_key="invoice:invoice-settlement",
        amount_paisa=300,
        user_id="user-1",
    )
    second = await service.settle_credit_application(
        payment,
        invoice_id="invoice-settlement",
        credit_application_key="invoice:invoice-settlement",
        amount_paisa=300,
        user_id="user-1",
    )

    assert isinstance(first, Success)
    assert isinstance(second, Success)
    assert (
        len(
            [
                call
                for call in provider.calls
                if call.operation.kind is ProviderOperationKind.CREATE_REFUND
            ]
        )
        == 1
    )
    assert payment.refund_amount == Decimal(3)
    assert payment.amount - int(payment.refund_amount * 100) == 700
    assert payment.amount - 300 == 700
