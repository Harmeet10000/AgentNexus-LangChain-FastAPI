"""Crash/retry contracts for chained provider intents."""

from asyncio import CancelledError
from datetime import UTC, datetime

import pytest
from returns.result import Success

from app.features.billing.providers import (
    CreatePlanOperation,
    InMemoryBillingProvider,
    ProviderOperationKind,
)
from app.features.billing.subscriptions.coordinator import (
    InMemoryLifecyclePersistence,
    LifecycleRecord,
    LifecycleRequest,
    SubscriptionLifecycleCoordinator,
)
from app.features.billing.subscriptions.lifecycle import (
    CommandMetadata,
    CreatedState,
    CreateSubscription,
    PolicyContext,
    SubscriptionSnapshot,
)


class _CrashAfterFirstMutation(InMemoryBillingProvider):
    def __init__(self, kind: ProviderOperationKind) -> None:
        super().__init__()
        self._crash_kind = kind
        self._crashed = False

    async def execute(self, operation):  # type: ignore[no-untyped-def]
        result = await super().execute(operation)
        if operation.kind is self._crash_kind and not self._crashed:
            self._crashed = True
            raise CancelledError
        return result


class _CrashAfterCustomerOutcome(InMemoryLifecyclePersistence):
    def __init__(self, records) -> None:  # type: ignore[no-untyped-def]
        super().__init__(records)
        self._crash = True

    async def provider_outcome(self, intent):  # type: ignore[no-untyped-def]
        result = await super().provider_outcome(intent)
        if (
            self._crash
            and intent.operation.kind is ProviderOperationKind.CREATE_CUSTOMER
            and intent.status.value == "succeeded"
        ):
            self._crash = False
            raise CancelledError
        return result


@pytest.mark.asyncio
async def test_customer_to_subscription_crash_has_one_post_per_child() -> None:
    persistence = InMemoryLifecyclePersistence(
        {
            "sub-1": LifecycleRecord(
                SubscriptionSnapshot("sub-1", "user-1", "plan-1", CreatedState()),
                customer_email="customer@example.test",
            )
        }
    )
    provider = _CrashAfterFirstMutation(ProviderOperationKind.CREATE_CUSTOMER)
    coordinator = SubscriptionLifecycleCoordinator(persistence, provider)
    request = LifecycleRequest(
        "sub-1",
        CreateSubscription(CommandMetadata("create-1"), "plan-1"),
        PolicyContext(now=datetime.now(tz=UTC)),
    )

    with pytest.raises(CancelledError):
        await coordinator.apply(request, "command-1")

    recovered = await coordinator.apply(request, "command-1")

    assert isinstance(recovered, Success)
    calls = [call.operation.kind for call in provider.calls]
    assert calls.count(ProviderOperationKind.CREATE_CUSTOMER) == 1
    assert calls.count(ProviderOperationKind.CREATE_SUBSCRIPTION) == 1
    assert calls.count(ProviderOperationKind.FIND_CUSTOMER_BY_EMAIL) == 1


@pytest.mark.asyncio
async def test_customer_outcome_crash_still_records_and_executes_child() -> None:
    persistence = _CrashAfterCustomerOutcome(
        {
            "sub-1": LifecycleRecord(
                SubscriptionSnapshot("sub-1", "user-1", "plan-1", CreatedState()),
                customer_email="customer@example.test",
            )
        }
    )
    provider = InMemoryBillingProvider()
    coordinator = SubscriptionLifecycleCoordinator(persistence, provider)
    request = LifecycleRequest(
        "sub-1",
        CreateSubscription(CommandMetadata("create-1"), "plan-1"),
        PolicyContext(now=datetime.now(tz=UTC)),
    )

    with pytest.raises(CancelledError):
        await coordinator.apply(request, "command-1")

    recovered = await coordinator.apply(request, "command-1")

    assert isinstance(recovered, Success)
    calls = [call.operation.kind for call in provider.calls]
    assert calls.count(ProviderOperationKind.CREATE_CUSTOMER) == 1
    assert calls.count(ProviderOperationKind.CREATE_SUBSCRIPTION) == 1


@pytest.mark.asyncio
async def test_plan_crash_recovery_reads_correlation_before_converging() -> None:
    provider = _CrashAfterFirstMutation(ProviderOperationKind.CREATE_PLAN)
    persistence = InMemoryLifecyclePersistence()
    coordinator = SubscriptionLifecycleCoordinator(persistence, provider)
    operation = CreatePlanOperation(
        name="Pro",
        amount=1000,
        currency="INR",
        period="monthly",
        interval=1,
        correlation_id="billing:plan:create:plan-1",
    )

    with pytest.raises(CancelledError):
        await coordinator.execute_provider_operation(
            operation,
            subscription_id=None,
            idempotency_key=operation.correlation_id,
        )

    recovered = await coordinator.recover_pending()

    assert isinstance(recovered, Success)
    calls = [call.operation.kind for call in provider.calls]
    assert calls.count(ProviderOperationKind.CREATE_PLAN) == 1
    assert calls.count(ProviderOperationKind.FIND_PLAN_BY_CORRELATION) == 1
