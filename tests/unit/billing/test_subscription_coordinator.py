"""Coordinator contract tests: callers know commands/events, never provider shapes."""

from dataclasses import replace
from datetime import UTC, datetime

import pytest
from returns.result import Failure, Success

from app.features.billing.providers import (
    CreatePaymentLinkOperation,
    CreateSubscriptionOperation,
    InMemoryBillingProvider,
    ProviderErrorCode,
    ProviderInvalidRequestError,
    ProviderManualReviewRequiredError,
    ProviderOperationKind,
    ProviderUnavailableError,
)
from app.features.billing.subscriptions.coordinator import (
    CoordinatorStatus,
    InMemoryLifecyclePersistence,
    IntentStatus,
    LifecycleCoordinatorProviderError,
    LifecycleCoordinatorVersionConflictError,
    LifecycleRecord,
    LifecycleRequest,
    ProviderOperationIntent,
    SubscriptionLifecycleCoordinator,
)
from app.features.billing.subscriptions.lifecycle import (
    ActiveState,
    AuthenticatedState,
    CancelSubscription,
    ChangeSubscriptionPlan,
    CommandMetadata,
    CreatedState,
    CreateSubscription,
    EventMetadata,
    PaymentCaptured,
    PendingUpgrade,
    PolicyContext,
    SubscriptionActivated,
    SubscriptionAuthenticated,
    SubscriptionSnapshot,
)

NOW = datetime(2026, 1, 15, tzinfo=UTC)
START = datetime(2026, 1, 1, tzinfo=UTC)
END = datetime(2026, 2, 1, tzinfo=UTC)


def _context() -> PolicyContext:
    return PolicyContext(now=NOW)


def _active_store() -> InMemoryLifecyclePersistence:
    snapshot = SubscriptionSnapshot(
        id="sub-1",
        user_id="user-1",
        plan_id="plan-1",
        state=ActiveState(START, END),
        version=3,
        observed_plan_id="plan-1",
    )
    return InMemoryLifecyclePersistence(
        {
            "sub-1": LifecycleRecord(
                snapshot,
                customer_email="user@example.test",
                provider_customer_id="cust-1",
                provider_subscription_id="sub_memory_1",
                provider_plan_id="plan-1",
            )
        }
    )


@pytest.mark.asyncio
async def test_create_authenticate_and_activate_workflow_preserves_entitlement_boundary() -> None:
    persistence = InMemoryLifecyclePersistence(
        {
            "sub-1": LifecycleRecord(
                SubscriptionSnapshot("sub-1", "user-1", "plan-1", CreatedState()),
                customer_email="user@example.test",
            )
        }
    )
    coordinator = SubscriptionLifecycleCoordinator(persistence, InMemoryBillingProvider())

    created = await coordinator.apply(
        LifecycleRequest(
            "sub-1",
            CreateSubscription(CommandMetadata("create-1"), "plan-1"),
            _context(),
        ),
        "command-1",
    )
    assert created.unwrap().grants_paid_entitlement is False
    assert created.unwrap().snapshot.state is not None

    authenticated = await coordinator.observe(
        LifecycleRequest(
            "sub-1",
            SubscriptionAuthenticated(EventMetadata("event-auth", NOW)),
            _context(),
        ),
        "event-auth",
    )
    assert isinstance(authenticated.unwrap().snapshot.state, AuthenticatedState)
    assert authenticated.unwrap().grants_paid_entitlement is False

    activated = await coordinator.observe(
        LifecycleRequest(
            "sub-1",
            SubscriptionActivated(EventMetadata("event-active", NOW), START, END),
            _context(),
        ),
        "event-active",
    )
    assert activated.unwrap().grants_paid_entitlement is True


@pytest.mark.asyncio
async def test_duplicate_command_and_event_are_no_provider_mutations() -> None:
    persistence = InMemoryLifecyclePersistence(
        {
            "sub-1": LifecycleRecord(
                SubscriptionSnapshot("sub-1", "user-1", "plan-1", CreatedState()),
                customer_email="user@example.test",
            )
        }
    )
    provider = InMemoryBillingProvider()
    coordinator = SubscriptionLifecycleCoordinator(persistence, provider)
    request = LifecycleRequest(
        "sub-1", CreateSubscription(CommandMetadata("create-1"), "plan-1"), _context()
    )

    first = await coordinator.apply(request, "same-key")
    calls = len(provider.calls)
    second = await coordinator.apply(request, "same-key")

    assert first.unwrap().snapshot == second.unwrap().snapshot
    assert second.unwrap().status is CoordinatorStatus.APPLIED
    assert len(provider.calls) == calls


@pytest.mark.asyncio
async def test_upgrade_stages_before_capture_and_converges_once_after_capture() -> None:
    persistence = _active_store()
    provider = InMemoryBillingProvider()
    # Seed the provider's known subscription through its public port.
    await provider.execute(CreateSubscriptionOperation("plan-1", "cust-1", 0, 1, True, "seed"))
    persistence.records["sub-1"] = LifecycleRecord(
        persistence.records["sub-1"].snapshot,
        provider_customer_id="cust-1",
        provider_subscription_id="sub_memory_1",
        provider_plan_id="plan-1",
    )
    coordinator = SubscriptionLifecycleCoordinator(persistence, provider)
    request = LifecycleRequest(
        "sub-1",
        ChangeSubscriptionPlan(CommandMetadata("change-1"), "plan-2", NOW),
        PolicyContext(
            now=NOW,
            current_plan_amount=1000,
            target_plan_amount=2000,
            target_currency="INR",
            target_provider_plan_id="plan-2",
        ),
    )

    staged = await coordinator.apply(request, "change-key")

    staged_snapshot = staged.unwrap().snapshot
    assert staged_snapshot.plan_id == "plan-2"
    assert staged_snapshot.observed_plan_id == "plan-1"
    assert staged_snapshot.grants_paid_entitlement is False
    assert staged_snapshot.pending_upgrade is not None
    pending = staged_snapshot.pending_upgrade
    assert persistence.records["sub-1"].snapshot.pending_upgrade == pending
    assert not any(
        call.operation.kind is ProviderOperationKind.CHANGE_SUBSCRIPTION_PLAN
        for call in provider.calls
    )

    captured = await coordinator.observe(
        PaymentCaptured(
            EventMetadata("capture-1", NOW),
            payment_id="payment-1",
            amount_paisa=pending.amount_paisa,
            currency=pending.currency,
        ),
        "capture-1",
        subscription_id="sub-1",
        context=_context(),
    )

    assert captured.unwrap().snapshot.pending_upgrade is None
    assert captured.unwrap().snapshot.observed_plan_id == "plan-2"
    assert [
        call.operation.kind
        for call in provider.calls
        if call.operation.kind is ProviderOperationKind.CHANGE_SUBSCRIPTION_PLAN
    ] == [ProviderOperationKind.CHANGE_SUBSCRIPTION_PLAN]


@pytest.mark.asyncio
async def test_scheduled_capture_reconciles_exact_pending_link_before_plan_change() -> None:
    persistence = _active_store()
    provider = InMemoryBillingProvider()
    await provider.execute(CreateSubscriptionOperation("plan-1", "cust-1", 0, 1, True, "seed"))
    persistence.records["sub-1"] = replace(
        persistence.records["sub-1"],
        provider_customer_id="cust-1",
        provider_subscription_id="sub_memory_1",
        provider_plan_id="plan-1",
    )
    coordinator = SubscriptionLifecycleCoordinator(persistence, provider)
    staged = await coordinator.apply(
        ChangeSubscriptionPlan(CommandMetadata("scheduled-upgrade"), "plan-2", NOW),
        "scheduled-upgrade",
        subscription_id="sub-1",
        context=PolicyContext(
            now=NOW,
            current_plan_amount=1_000,
            target_plan_amount=2_000,
            target_currency="INR",
            target_provider_plan_id="provider-plan-2",
        ),
    )
    assert isinstance(staged, Success)
    pending = staged.unwrap().snapshot.pending_upgrade
    assert pending is not None
    target = persistence.intents[pending.payment_link_key]
    persistence.intents[pending.payment_link_key] = replace(target, status=IntentStatus.PENDING)
    sibling_key = "unrelated-payment-link"
    persistence.intents[sibling_key] = ProviderOperationIntent(
        key=sibling_key,
        subscription_id="sub-1",
        operation=CreatePaymentLinkOperation(
            amount=pending.amount_paisa,
            currency=pending.currency,
            description="unrelated",
            correlation_id=sibling_key,
            notes={
                "subscription_id": "sub-1",
                "upgrade_dependency": "true",
                "target_plan_id": "sibling-plan",
                "target_provider_plan_id": "sibling-provider-plan",
                "amount_paisa": str(pending.amount_paisa),
                "currency": pending.currency,
            },
        ),
    )

    captured = await coordinator.observe(
        PaymentCaptured(
            EventMetadata("scheduled-capture", NOW),
            payment_id="payment-scheduled",
            amount_paisa=pending.amount_paisa,
            currency=pending.currency,
        ),
        "scheduled-capture",
        subscription_id="sub-1",
        context=_context(),
    )

    assert isinstance(captured, Success)
    assert persistence.intents[pending.payment_link_key].status is IntentStatus.SUCCEEDED
    assert persistence.intents[sibling_key].status is IntentStatus.PENDING
    assert [
        call.operation.correlation_id
        for call in provider.calls
        if call.operation.kind is ProviderOperationKind.FIND_PAYMENT_LINK_BY_CORRELATION
    ] == [target.operation.correlation_id]
    assert [
        call.operation.kind
        for call in provider.calls
        if call.operation.kind is ProviderOperationKind.CHANGE_SUBSCRIPTION_PLAN
    ] == [ProviderOperationKind.CHANGE_SUBSCRIPTION_PLAN]


@pytest.mark.asyncio
async def test_scheduled_capture_unknown_link_leaves_pending_entitlement_unchanged() -> None:
    persistence = _active_store()
    pending = PendingUpgrade(
        payment_link_key="missing-payment-link",
        target_plan_id="plan-2",
        target_provider_plan_id="provider-plan-2",
        amount_paisa=1_000,
        currency="INR",
    )
    before = persistence.records["sub-1"].snapshot
    persistence.records["sub-1"] = replace(
        persistence.records["sub-1"], snapshot=replace(before, pending_upgrade=pending)
    )
    persistence.intents[pending.payment_link_key] = ProviderOperationIntent(
        key=pending.payment_link_key,
        subscription_id="sub-1",
        operation=CreatePaymentLinkOperation(
            amount=pending.amount_paisa,
            currency=pending.currency,
            description="upgrade",
            correlation_id=pending.payment_link_key,
            notes={
                "subscription_id": "sub-1",
                "upgrade_dependency": "true",
                "target_plan_id": pending.target_plan_id,
                "target_provider_plan_id": pending.target_provider_plan_id,
                "amount_paisa": str(pending.amount_paisa),
                "currency": pending.currency,
            },
        ),
    )
    # Model a provider POST that may have happened before the worker died.
    persistence._claimed_intents.add(pending.payment_link_key)
    provider = InMemoryBillingProvider()
    coordinator = SubscriptionLifecycleCoordinator(persistence, provider)

    captured = await coordinator.observe(
        PaymentCaptured(
            EventMetadata("unknown-capture", NOW),
            payment_id="payment-unknown",
            amount_paisa=pending.amount_paisa,
            currency=pending.currency,
        ),
        "unknown-capture",
        subscription_id="sub-1",
        context=_context(),
    )

    assert isinstance(captured, Failure)
    assert persistence.records["sub-1"].snapshot == replace(before, pending_upgrade=pending)
    assert not any(
        call.operation.kind is ProviderOperationKind.CHANGE_SUBSCRIPTION_PLAN
        for call in provider.calls
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "code"),
    [
        (ProviderInvalidRequestError(), ProviderErrorCode.INVALID_REQUEST),
        (ProviderUnavailableError(), ProviderErrorCode.UNAVAILABLE),
        (
            ProviderManualReviewRequiredError(
                ProviderOperationKind.CHANGE_SUBSCRIPTION_PLAN, "manual"
            ),
            ProviderErrorCode.MANUAL_REVIEW_REQUIRED,
        ),
    ],
)
async def test_provider_failures_are_typed_and_not_hidden(
    error: object, code: ProviderErrorCode
) -> None:
    persistence = _active_store()
    provider = InMemoryBillingProvider(
        {ProviderOperationKind.CANCEL_SUBSCRIPTION: (Failure(error),)}  # type: ignore[arg-type]
    )
    coordinator = SubscriptionLifecycleCoordinator(provider=provider, persistence=persistence)
    result = await coordinator.apply(
        LifecycleRequest(
            "sub-1",
            CancelSubscription(CommandMetadata("cancel-1"), cancel_at_period_end=False),
            _context(),
        ),
        "cancel-key",
    )

    assert isinstance(result, Failure)
    assert isinstance(result.failure(), LifecycleCoordinatorProviderError)
    assert result.failure().provider_code is code


@pytest.mark.asyncio
async def test_optimistic_conflict_is_returned_before_provider_mutation() -> None:
    persistence = _active_store()
    provider = InMemoryBillingProvider()
    coordinator = SubscriptionLifecycleCoordinator(persistence, provider)
    result = await coordinator.apply(
        LifecycleRequest(
            "sub-1",
            CancelSubscription(
                CommandMetadata("stale", expected_version=2), cancel_at_period_end=True
            ),
            _context(),
        ),
        "stale-key",
    )

    assert isinstance(result, Failure)
    assert isinstance(result.failure(), LifecycleCoordinatorVersionConflictError)
    assert provider.calls == ()


@pytest.mark.asyncio
async def test_effect_translation_gates_remote_plan_until_matching_capture() -> None:
    persistence = _active_store()
    provider = InMemoryBillingProvider()
    await provider.execute(CreateSubscriptionOperation("plan-1", "cust-1", 0, 1, True, "seed"))
    coordinator = SubscriptionLifecycleCoordinator(persistence, provider)
    result = await coordinator.apply(
        LifecycleRequest(
            "sub-1",
            ChangeSubscriptionPlan(CommandMetadata("change-1"), "plan-2", NOW),
            PolicyContext(
                now=NOW,
                current_plan_amount=1000,
                target_plan_amount=2000,
                target_currency="INR",
                target_provider_plan_id="plan-2",
            ),
        ),
        "change-key",
    )

    result_snapshot = result.unwrap().snapshot
    assert result_snapshot.plan_id == "plan-2"
    # The desired plan changes immediately, while the provider observation
    # remains old until the matching payment is captured.
    assert result_snapshot.observed_plan_id == "plan-1"
    assert result_snapshot.grants_paid_entitlement is False
    pending = result_snapshot.pending_upgrade
    assert pending is not None
    assert persistence.records["sub-1"].snapshot.pending_upgrade == pending
    assert not any(
        call.operation.kind is ProviderOperationKind.CHANGE_SUBSCRIPTION_PLAN
        for call in provider.calls
    )

    captured = await coordinator.observe(
        PaymentCaptured(
            EventMetadata("capture-2", NOW),
            payment_id="payment-2",
            amount_paisa=pending.amount_paisa,
            currency=pending.currency,
        ),
        "capture-2",
        subscription_id="sub-1",
        context=_context(),
    )

    assert captured.unwrap().snapshot.observed_plan_id == "plan-2"
    assert captured.unwrap().snapshot.pending_upgrade is None
    assert [
        call.operation.kind
        for call in provider.calls
        if call.operation.kind is ProviderOperationKind.CHANGE_SUBSCRIPTION_PLAN
    ] == [ProviderOperationKind.CHANGE_SUBSCRIPTION_PLAN]
