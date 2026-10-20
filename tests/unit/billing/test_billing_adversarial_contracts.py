"""Black-box contract tests for billing lifecycle and provider boundaries.

These tests are executable specifications: lifecycle intent is asserted
separately from observed provider state, and provider failures are asserted
through the closed ``Result`` error contract.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, get_args

import pytest
from returns.result import Failure, Success

from app.features.billing.payments.clients.razorpay_client import RazorpayRetryableError
from app.features.billing.providers import (
    CancelSubscriptionOperation,
    CreatePaymentLinkOperation,
    CreateRefundOperation,
    CreateSubscriptionOperation,
    FindPaymentLinkByCorrelationOperation,
    FindRefundByCorrelationOperation,
    FindSubscriptionByCorrelationOperation,
    InMemoryBillingProvider,
    ObserveSubscriptionOperation,
    PauseSubscriptionOperation,
    ProviderAuthenticationError,
    ProviderError,
    ProviderErrorCode,
    ProviderInvalidRequestError,
    ProviderManualReviewRequiredError,
    ProviderNotFoundError,
    ProviderOperationKind,
    ProviderRateLimitedError,
    ProviderReconciliationRequiredError,
    ProviderUnavailableError,
    ProviderUnexpectedError,
    RazorpayBillingProvider,
    ResumeSubscriptionOperation,
)
from app.features.billing.providers.protocol import (
    ChangeSubscriptionPlanOperation,
    CreateCustomerOperation,
    ProviderCustomer,
    ProviderOutcomeUnknownError,
    ProviderSubscriptionState,
    ProviderSuccess,
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
    HaltedState,
    PastDueState,
    PausedState,
    PauseSubscription,
    PaymentCaptured,
    PaymentFailed,
    PolicyContext,
    RemoteSyncState,
    SubscriptionActivated,
    SubscriptionAuthenticated,
    SubscriptionCancelled,
    SubscriptionCharged,
    SubscriptionHalted,
    SubscriptionMachine,
    SubscriptionPaused,
    SubscriptionResumed,
    SubscriptionSnapshot,
    TransitionOutcome,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any


NOW = datetime(2026, 1, 15, 12, tzinfo=UTC)
NAIVE_NOW = NOW.replace(tzinfo=None)
START = datetime(2026, 1, 1, tzinfo=UTC)
END = datetime(2026, 2, 1, tzinfo=UTC)


def _snapshot(state: object, *, version: int = 7) -> SubscriptionSnapshot:
    return SubscriptionSnapshot(
        id="sub-contract",
        user_id="user-contract",
        plan_id="plan-old",
        state=state,  # type: ignore[arg-type]
        version=version,
    )


def _command(request_id: str = "request-contract") -> CommandMetadata:
    return CommandMetadata(request_id=request_id)


def _event(event_id: str, *, replay: bool = False) -> EventMetadata:
    return EventMetadata(event_id=event_id, occurred_at=NOW, replay=replay)


def _context(**kwargs: Any) -> PolicyContext:
    return PolicyContext(now=NOW, **kwargs)


def test_period_end_cancel_keeps_entitlement_until_period_end() -> None:
    """Requirement 5.7: period-end cancellation is an intent, not termination."""

    before = _snapshot(ActiveState(START, END))
    result = (
        SubscriptionMachine()
        .plan(
            before,
            CancelSubscription(_command(), cancel_at_period_end=True),
            _context(),
        )
        .unwrap()
    )

    assert isinstance(result.after.state, ActiveState)
    assert result.after.grants_paid_entitlement is True
    assert result.after.remote_sync is RemoteSyncState.PENDING
    assert any(dict(effect.values).get("cancel_at_period_end") for effect in result.effects)


def test_paused_subscription_does_not_grant_paid_entitlement() -> None:
    assert _snapshot(PausedState(start=NOW)).grants_paid_entitlement is False


@pytest.mark.parametrize("state", [PausedState(start=NOW), HaltedState()])
def test_payment_capture_cannot_reactivate_paused_or_halted_subscription(state: object) -> None:
    before = _snapshot(state)
    result = (
        SubscriptionMachine()
        .plan(
            before,
            PaymentCaptured(_event("payment-stale"), payment_id="pay-stale", amount_paisa=100),
            _context(),
        )
        .unwrap()
    )

    assert result.outcome is TransitionOutcome.SKIP
    assert result.after == before
    assert result.effects == ()


@pytest.mark.parametrize(
    ("event", "expected_state"),
    [
        (
            SubscriptionCharged(_event("charge-omitted")),
            ActiveState(START, END),
        ),
        (
            SubscriptionActivated(_event("activation-omitted")),
            ActiveState(START, END),
        ),
    ],
)
def test_omitted_period_facts_do_not_erase_current_period(
    event: object, expected_state: object
) -> None:
    before = _snapshot(ActiveState(START, END))
    result = SubscriptionMachine().plan(before, event, _context()).unwrap()  # type: ignore[arg-type]

    assert result.after.state == expected_state


def test_stale_period_facts_do_not_regress_current_period() -> None:
    before = _snapshot(ActiveState(START, END))
    result = (
        SubscriptionMachine()
        .plan(
            before,
            SubscriptionCharged(
                _event("charge-old"),
                period_start=START - timedelta(days=31),
                period_end=END - timedelta(days=31),
            ),
            _context(),
        )
        .unwrap()
    )

    assert result.after.state == before.state
    assert result.outcome is TransitionOutcome.SKIP


@pytest.mark.parametrize(
    "event_factory",
    [
        lambda: SubscriptionAuthenticated(_event("replay-auth", replay=True)),
        lambda: SubscriptionActivated(_event("replay-active", replay=True), START, END),
        lambda: SubscriptionCharged(_event("replay-charge", replay=True), START, END),
        lambda: PaymentCaptured(_event("replay-capture", replay=True), "pay-1", 100),
        lambda: PaymentFailed(_event("replay-failed", replay=True), "pay-2"),
        lambda: SubscriptionPaused(_event("replay-paused", replay=True)),
        lambda: SubscriptionResumed(_event("replay-resumed", replay=True)),
        lambda: SubscriptionHalted(_event("replay-halted", replay=True)),
        lambda: SubscriptionCancelled(_event("replay-cancelled", replay=True)),
    ],
)
def test_every_replayed_event_is_a_no_effect_skip(event_factory: Callable[[], object]) -> None:
    event = event_factory()
    state = (
        PausedState(start=NOW)
        if isinstance(event, SubscriptionResumed)
        else PastDueState()
        if isinstance(event, SubscriptionHalted)
        else CreatedState()
        if isinstance(event, SubscriptionAuthenticated)
        else AuthenticatedState()
        if isinstance(event, SubscriptionActivated)
        else ActiveState(START, END)
    )
    before = _snapshot(state)
    result = SubscriptionMachine().plan(before, event, _context()).unwrap()  # type: ignore[arg-type]

    assert result.outcome is TransitionOutcome.SKIP
    assert result.after == before
    assert result.effects == ()


def test_no_op_does_not_bump_optimistic_version() -> None:
    before = _snapshot(ActiveState(START, END), version=11)
    result = (
        SubscriptionMachine()
        .plan(
            before,
            SubscriptionAuthenticated(_event("already-authenticated")),
            _context(),
        )
        .unwrap()
    )

    assert result.outcome is TransitionOutcome.NO_OP
    assert result.after.version == before.version


def test_context_requires_aware_time() -> None:
    result = SubscriptionMachine().plan(
        _snapshot(ActiveState(START, END)),
        SubscriptionAuthenticated(_event("aware-time")),
        PolicyContext(now=NAIVE_NOW),
    )

    assert isinstance(result, Failure)


@pytest.mark.parametrize("duration_days", [0, -1, 366])
def test_pause_duration_is_bounded(duration_days: int) -> None:
    result = SubscriptionMachine().plan(
        _snapshot(ActiveState(START, END)),
        PauseSubscription(_command(), duration_days=duration_days),
        _context(),
    )

    assert isinstance(result, Failure)


def test_create_has_pending_remote_sync_and_audit_intent() -> None:
    result = (
        SubscriptionMachine()
        .plan(
            _snapshot(CreatedState()),
            CreateSubscription(_command(), plan_id="plan-old"),
            _context(),
        )
        .unwrap()
    )

    assert result.after.remote_sync is RemoteSyncState.PENDING
    assert any(effect.kind.value == "audit" for effect in result.effects)


def test_plan_change_rejects_effective_date_after_period_end() -> None:
    result = SubscriptionMachine().plan(
        _snapshot(ActiveState(START, END)),
        ChangeSubscriptionPlan(_command(), "plan-new", END + timedelta(seconds=1)),
        _context(),
    )

    assert isinstance(result, Failure)


def test_plan_change_rejects_partial_financial_facts() -> None:
    result = SubscriptionMachine().plan(
        _snapshot(ActiveState(START, END)),
        ChangeSubscriptionPlan(_command(), "plan-new", NOW),
        _context(current_plan_amount=1000),
    )

    assert isinstance(result, Failure)


def test_plan_change_updates_desired_without_advancing_observed_plan() -> None:
    before = SubscriptionSnapshot(
        id="sub-contract",
        user_id="user-contract",
        plan_id="plan-old",
        state=ActiveState(START, END),
        observed_plan_id="plan-old",
    )
    result = (
        SubscriptionMachine()
        .plan(
            before,
            ChangeSubscriptionPlan(_command(), "plan-new", NOW),
            _context(current_plan_amount=1000, target_plan_amount=2000),
        )
        .unwrap()
    )

    assert result.after.plan_id == "plan-new"
    assert result.after.observed_plan_id == "plan-old"
    assert result.after.remote_sync is RemoteSyncState.PENDING


@pytest.mark.asyncio
async def test_provider_execute_uses_result_boundary() -> None:
    result = await InMemoryBillingProvider().execute(
        CreateSubscriptionOperation(
            plan_id="plan-old",
            customer_id="customer-1",
            total_count=12,
            quantity=1,
            customer_notify=True,
            correlation_id="result-boundary",
        )
    )

    from returns.result import Result

    assert isinstance(result, Result)


def test_unknown_provider_outcome_advertises_reconciliation_postcondition() -> None:
    assert ProviderOutcomeUnknownError.requires_reconciliation is True


@pytest.mark.asyncio
async def test_ambiguous_mutation_is_attempted_once_before_reconciliation() -> None:
    class AmbiguousRazorpay:
        def __init__(self) -> None:
            self.attempts = 0

        async def create_subscription(self, **kwargs: Any) -> dict[str, Any]:
            del kwargs
            self.attempts += 1
            raise RazorpayRetryableError(service="razorpay", detail="timeout")

    client = AmbiguousRazorpay()
    operation = CreateSubscriptionOperation(
        plan_id="plan-old",
        customer_id="customer-1",
        total_count=12,
        quantity=1,
        customer_notify=True,
        correlation_id="ambiguous-once",
    )

    result = await RazorpayBillingProvider(client).execute(operation)

    assert isinstance(result, Failure)
    assert isinstance(result.failure(), ProviderOutcomeUnknownError)
    assert client.attempts == 1


def test_provider_port_has_customer_and_change_plan_operations() -> None:
    from app.features.billing.providers import ProviderOperationKind

    values = {kind.value for kind in ProviderOperationKind}
    assert "create_customer" in values
    assert "change_subscription_plan" in values


@pytest.mark.asyncio
async def test_script_double_contract_is_fifo_then_deterministic_default() -> None:
    operation = CreateSubscriptionOperation(
        plan_id="plan-old",
        customer_id="customer-contract",
        total_count=12,
        quantity=1,
        customer_notify=True,
        correlation_id="create-contract",
    )
    provider = InMemoryBillingProvider()

    first = await provider.execute(operation)
    second = await provider.execute(operation)

    assert isinstance(first, Success)
    assert isinstance(second, Success)
    assert isinstance(first.unwrap(), ProviderSuccess)
    assert isinstance(second.unwrap(), ProviderSuccess)
    assert (
        first.unwrap().correlation_id == second.unwrap().correlation_id == operation.correlation_id
    )
    assert first.unwrap().value == second.unwrap().value
    assert len(provider.calls) == 2


@pytest.mark.asyncio
async def test_stateful_double_supports_customer_and_change_plan_operations() -> None:
    provider = InMemoryBillingProvider()
    customer = await provider.execute(
        CreateCustomerOperation(
            email="customer@example.test",
            contact=None,
            name="Contract Customer",
            correlation_id="customer-contract",
        )
    )

    assert isinstance(customer, Success)
    assert isinstance(customer.unwrap().value, ProviderCustomer)

    created = await provider.execute(
        CreateSubscriptionOperation(
            plan_id="plan-old",
            customer_id=customer.unwrap().value.provider_customer_id,
            total_count=12,
            quantity=1,
            customer_notify=True,
            correlation_id="subscription-contract",
        )
    )
    assert isinstance(created, Success)
    assert isinstance(created.unwrap().value, ProviderSubscriptionState)
    subscription_id = created.unwrap().value.provider_subscription_id

    changed = await provider.execute(
        ChangeSubscriptionPlanOperation(
            provider_subscription_id=subscription_id,
            plan_id="plan-new",
            correlation_id="change-plan-contract",
        )
    )

    assert isinstance(changed, Success)
    assert isinstance(changed.unwrap().value, ProviderSubscriptionState)
    assert changed.unwrap().value.provider_subscription_id == subscription_id


def test_provider_error_contract_is_closed_and_metadata_is_typed() -> None:
    expected_errors = {
        ProviderInvalidRequestError,
        ProviderAuthenticationError,
        ProviderRateLimitedError,
        ProviderUnavailableError,
        ProviderNotFoundError,
        ProviderOutcomeUnknownError,
        ProviderManualReviewRequiredError,
        ProviderReconciliationRequiredError,
        ProviderUnexpectedError,
    }
    # PEP 695 aliases expose their closed value through ``__value__`` at
    # runtime; checking it here prevents an adapter from growing an ad-hoc
    # error arm that callers cannot exhaustively handle.
    assert set(get_args(ProviderError.__value__)) == expected_errors

    for error_type in expected_errors:
        error = error_type(
            operation=ProviderOperationKind.CREATE_SUBSCRIPTION,
            correlation_id="typed-error",
        )
        assert isinstance(error.code, ProviderErrorCode)
        assert isinstance(error.retryable, bool)


@pytest.mark.asyncio
async def test_every_ambiguous_mutation_is_attempted_once() -> None:
    class AmbiguousRazorpay:
        def __init__(self) -> None:
            self.attempts: dict[str, int] = {}

        def __getattr__(self, name: str) -> Any:
            async def fail(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
                self.attempts[name] = self.attempts.get(name, 0) + 1
                raise RazorpayRetryableError(service="razorpay", detail="timeout")

            return fail

    operations = (
        CreateCustomerOperation("customer@example.test", None, None, "ambiguous-customer"),
        CreateSubscriptionOperation("plan-1", "customer-1", 12, 1, True, "ambiguous-create"),
        CancelSubscriptionOperation("sub-1", False, "ambiguous-cancel"),
        PauseSubscriptionOperation("sub-1", "ambiguous-pause"),
        ResumeSubscriptionOperation("sub-1", "ambiguous-resume"),
        ChangeSubscriptionPlanOperation("sub-1", "plan-2", "ambiguous-change"),
        CreatePaymentLinkOperation(500, "INR", "upgrade", "ambiguous-link"),
        CreateRefundOperation("pay-1", 500, "ambiguous-refund"),
    )
    client = AmbiguousRazorpay()
    provider = RazorpayBillingProvider(client)

    for operation in operations:
        result = await provider.execute(operation)
        assert isinstance(result, Failure)
        assert isinstance(result.failure(), ProviderOutcomeUnknownError)
        assert result.failure().operation is operation.kind
        assert result.failure().correlation_id == operation.correlation_id

    assert client.attempts == {
        "create_customer": 1,
        "create_subscription": 1,
        "cancel_subscription": 1,
        "pause_subscription": 1,
        "resume_subscription": 1,
        "change_subscription_plan": 1,
        "create_payment_link": 1,
        "create_refund": 1,
    }


@pytest.mark.asyncio
async def test_customer_provisioning_preserves_identity_and_correlation() -> None:
    class CustomerRazorpay:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def create_customer(self, **kwargs: Any) -> dict[str, Any]:
            self.calls.append(kwargs)
            return {"id": "cust-remote-1", "email": kwargs["email"]}

    client = CustomerRazorpay()
    operation = CreateCustomerOperation(
        email="customer@example.test",
        contact="+919999999999",
        name="Contract Customer",
        correlation_id="customer-provisioning",
        notes={"tenant": "acme"},
    )
    result = await RazorpayBillingProvider(client).execute(operation)

    assert isinstance(result, Success)
    assert result.unwrap().operation is operation.kind
    assert result.unwrap().correlation_id == operation.correlation_id
    assert result.unwrap().value == ProviderCustomer("cust-remote-1", operation.email)
    assert client.calls == [
        {
            "email": operation.email,
            "contact": operation.contact,
            "name": operation.name,
            "notes": {"tenant": "acme", "billing_operation_id": operation.correlation_id},
        }
    ]


@pytest.mark.asyncio
async def test_remote_plan_change_preserves_postconditions_and_request_shape() -> None:
    class PlanChangeRazorpay:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, Any]]] = []

        async def change_subscription_plan(
            self, subscription_id: str, **kwargs: Any
        ) -> dict[str, Any]:
            self.calls.append((subscription_id, kwargs))
            return {"id": subscription_id, "status": "active"}

    client = PlanChangeRazorpay()
    operation = ChangeSubscriptionPlanOperation(
        provider_subscription_id="sub-remote-1",
        plan_id="plan-new",
        correlation_id="change-plan-remote",
        schedule_at="cycle_end",
    )
    result = await RazorpayBillingProvider(client).execute(operation)

    assert isinstance(result, Success)
    assert result.unwrap().operation is operation.kind
    assert result.unwrap().correlation_id == operation.correlation_id
    assert result.unwrap().value == ProviderSubscriptionState(
        provider_subscription_id="sub-remote-1", status="active"
    )
    assert client.calls == [("sub-remote-1", {"plan_id": "plan-new", "schedule_at": "cycle_end"})]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operation", "expected_error"),
    [
        (
            CreatePaymentLinkOperation(500, "INR", "upgrade", "unknown-link"),
            ProviderOutcomeUnknownError,
        ),
        (CreateRefundOperation("pay-1", 500, "unknown-refund"), ProviderOutcomeUnknownError),
    ],
)
async def test_malformed_mutation_results_require_reconciliation(
    operation: object, expected_error: type[object]
) -> None:
    class MalformedRazorpay:
        async def create_payment_link(self, **_kwargs: Any) -> dict[str, Any]:
            return {"short_url": "https://example.test/missing-id"}

        async def create_refund(self, **_kwargs: Any) -> dict[str, Any]:
            return {"id": "refund-1", "payment_id": "other-payment", "amount": 1}

    result = await RazorpayBillingProvider(MalformedRazorpay()).execute(operation)  # type: ignore[arg-type]

    assert isinstance(result, Failure)
    assert isinstance(result.failure(), expected_error)
    assert result.failure().requires_reconciliation is True


@pytest.mark.asyncio
async def test_unresolved_correlation_searches_explicitly_require_manual_review() -> None:
    provider = InMemoryBillingProvider()
    searches = (
        FindSubscriptionByCorrelationOperation("missing-subscription"),
        FindPaymentLinkByCorrelationOperation("missing-link"),
        FindRefundByCorrelationOperation("pay-1", "missing-refund"),
    )

    for operation in searches:
        result = await provider.execute(operation)
        assert isinstance(result, Failure)
        assert isinstance(result.failure(), ProviderManualReviewRequiredError)
        assert result.failure().code is ProviderErrorCode.MANUAL_REVIEW_REQUIRED
        assert result.failure().correlation_id == operation.correlation_id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "expected_error"),
    [
        (401, ProviderAuthenticationError),
        (403, ProviderAuthenticationError),
        (404, ProviderNotFoundError),
        (422, ProviderInvalidRequestError),
    ],
)
async def test_permanent_provider_statuses_have_stable_error_taxonomy(
    status_code: int, expected_error: type[object]
) -> None:
    class RejectedRazorpay:
        async def cancel_subscription(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            from app.features.billing.payments.clients.razorpay_client import (
                RazorpayPermanentError,
            )

            raise RazorpayPermanentError(
                service="razorpay", detail="private provider detail", status_code=status_code
            )

    result = await RazorpayBillingProvider(RejectedRazorpay()).execute(
        CancelSubscriptionOperation("sub-1", False, f"status-{status_code}")
    )

    assert isinstance(result, Failure)
    assert isinstance(result.failure(), expected_error)
    assert result.failure().retryable is False
    assert "private provider detail" not in result.failure().message


@pytest.mark.asyncio
async def test_in_memory_reuses_correlation_identity_across_mutation_replays() -> None:
    provider = InMemoryBillingProvider()
    customer_operation = CreateCustomerOperation(
        "customer@example.test", None, "Customer", "customer-replay"
    )
    first_customer = await provider.execute(customer_operation)
    second_customer = await provider.execute(
        CreateCustomerOperation("CUSTOMER@EXAMPLE.TEST", None, "Customer", "customer-replay-2")
    )
    assert isinstance(first_customer, Success)
    assert isinstance(second_customer, Success)
    assert (
        first_customer.unwrap().value.provider_customer_id
        == second_customer.unwrap().value.provider_customer_id
    )

    subscription_operation = CreateSubscriptionOperation(
        "plan-1", first_customer.unwrap().value.provider_customer_id, 12, 1, True, "sub-replay"
    )
    first_subscription = await provider.execute(subscription_operation)
    second_subscription = await provider.execute(subscription_operation)
    assert isinstance(first_subscription, Success)
    assert isinstance(second_subscription, Success)
    assert first_subscription.unwrap().value == second_subscription.unwrap().value

    link_operation = CreatePaymentLinkOperation(500, "INR", "upgrade", "link-replay")
    first_link = await provider.execute(link_operation)
    second_link = await provider.execute(link_operation)
    refund_operation = CreateRefundOperation("pay-1", 500, "refund-replay")
    first_refund = await provider.execute(refund_operation)
    second_refund = await provider.execute(refund_operation)

    assert isinstance(first_link, Success)
    assert isinstance(second_link, Success)
    assert first_link.unwrap().value == second_link.unwrap().value
    assert isinstance(first_refund, Success)
    assert isinstance(second_refund, Success)
    assert first_refund.unwrap().value == second_refund.unwrap().value


@pytest.mark.asyncio
async def test_observe_malformed_read_requires_reconciliation() -> None:
    class MalformedReadRazorpay:
        async def fetch_subscription(self, _subscription_id: str) -> dict[str, Any]:
            return {"id": "sub-1", "status": "provider-added-future-status"}

    result = await RazorpayBillingProvider(MalformedReadRazorpay()).execute(
        ObserveSubscriptionOperation("sub-1", "observe-malformed")
    )

    assert isinstance(result, Failure)
    assert isinstance(result.failure(), ProviderReconciliationRequiredError)
    assert result.failure().code is ProviderErrorCode.RECONCILIATION_REQUIRED
