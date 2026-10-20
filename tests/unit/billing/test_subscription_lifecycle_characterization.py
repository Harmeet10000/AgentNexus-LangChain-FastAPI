"""Behavioral characterization for the subscription and payment seams.

These tests deliberately exercise public repository/service interfaces.  They
are migration surveillance for the contracts in ``.kiro``; they do not make
private helper behavior part of the contract.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from returns.result import Failure, Success

from app.features.billing.payments.clients.razorpay_client import (
    RazorpayPermanentError,
    RazorpayRetryableError,
)
from app.features.billing.payments.service import PaymentService
from app.features.billing.subscriptions.models import SubscriptionStatus
from app.features.billing.subscriptions.proration import calculate_plan_change_proration
from app.features.billing.subscriptions.repository import SubscriptionRepository
from app.features.billing.webhooks.exceptions import WebhookCollaboratorError
from app.features.billing.webhooks.models import WebhookEventStatus, WebhookEventType
from app.features.billing.webhooks.service import WebhookService

_TRANSITIONS: dict[SubscriptionStatus, set[SubscriptionStatus]] = {
    SubscriptionStatus.CREATED: {SubscriptionStatus.AUTHENTICATED},
    SubscriptionStatus.AUTHENTICATED: {SubscriptionStatus.ACTIVE},
    SubscriptionStatus.ACTIVE: {
        SubscriptionStatus.PAST_DUE,
        SubscriptionStatus.PAUSED,
        SubscriptionStatus.CANCELLED,
    },
    SubscriptionStatus.PAST_DUE: {SubscriptionStatus.ACTIVE, SubscriptionStatus.HALTED},
    SubscriptionStatus.HALTED: {SubscriptionStatus.ACTIVE, SubscriptionStatus.CANCELLED},
    SubscriptionStatus.PAUSED: {SubscriptionStatus.ACTIVE, SubscriptionStatus.CANCELLED},
    SubscriptionStatus.CANCELLED: set(),
    SubscriptionStatus.EXPIRED: set(),
}


def _db_result(value: object) -> SimpleNamespace:
    return SimpleNamespace(scalar_one_or_none=lambda: value)


def _subscription(*, status: str, version: int = 4) -> SimpleNamespace:
    return SimpleNamespace(
        id="subscription-1",
        user_id="user-1",
        plan_id="plan-1",
        status=status,
        version=version,
    )


@pytest.mark.parametrize(
    ("current", "target", "allowed"),
    [
        (current, target, target in targets)
        for current, targets in _TRANSITIONS.items()
        for target in SubscriptionStatus
    ],
)
@pytest.mark.asyncio
async def test_subscription_repository_matches_specified_transition_matrix(
    current: SubscriptionStatus,
    target: SubscriptionStatus,
    allowed: bool,
) -> None:
    """Requirements 4.1-4.7: terminal states and every edge are covered."""
    session = SimpleNamespace(
        execute=AsyncMock(return_value=_db_result(_subscription(status=target.value)))
    )
    repository = SubscriptionRepository(session)

    result = await repository.update_status(
        _subscription(status=current.value),
        target,
        expected_version=4,
    )

    assert isinstance(result, Success) is allowed
    if allowed:
        session.execute.assert_awaited_once()
    else:
        session.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_subscription_repository_reports_optimistic_version_conflict() -> None:
    """Requirement 29: a stale writer receives a typed conflict."""
    session = SimpleNamespace(execute=AsyncMock(return_value=_db_result(None)))
    repository = SubscriptionRepository(session)

    result = await repository.update_with_lock(
        _subscription(status=SubscriptionStatus.ACTIVE.value, version=7),
        expected_version=7,
        values={"retry_count": 1},
    )

    assert isinstance(result, Failure)
    assert result.failure().expected_version == 7
    assert result.failure().subscription_id == "subscription-1"


def _proration_subscription(*, start: datetime, end: datetime) -> SimpleNamespace:
    return SimpleNamespace(
        id="subscription-1",
        status=SubscriptionStatus.ACTIVE,
        current_period_start=start,
        current_period_end=end,
    )


def _plan(*, plan_id: str, amount: int) -> SimpleNamespace:
    return SimpleNamespace(
        id=plan_id,
        amount=amount,
        interval="monthly",
        tax_rate=Decimal("0.18"),
    )


@pytest.mark.parametrize(
    ("effective_at", "expected_fraction"),
    [
        (datetime(2026, 1, 1, tzinfo=UTC), Decimal(1)),
        (datetime(2026, 1, 31, tzinfo=UTC), Decimal(0)),
    ],
)
def test_proration_period_boundaries_are_deterministic(
    effective_at: datetime,
    expected_fraction: Decimal,
) -> None:
    """Requirements 5.1, 5.6, and 33: period edges use exact arithmetic."""
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 31, tzinfo=UTC)

    result = calculate_plan_change_proration(
        _proration_subscription(start=start, end=end),
        _plan(plan_id="current", amount=10000),
        _plan(plan_id="new", amount=20000),
        effective_date=effective_at,
    )

    assert isinstance(result, Success)
    assert result.unwrap().remaining_fraction == expected_fraction


def test_proration_rejects_dates_after_the_billing_period() -> None:
    """Requirement 5.5: effective dates must be inside the current period."""
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 31, tzinfo=UTC)

    result = calculate_plan_change_proration(
        _proration_subscription(start=start, end=end),
        _plan(plan_id="current", amount=10000),
        _plan(plan_id="new", amount=20000),
        effective_date=datetime(2026, 2, 1, tzinfo=UTC),
    )

    assert isinstance(result, Failure)


def test_proration_rejects_dates_before_the_billing_period() -> None:
    """Requirement 5.5: effective dates must be inside the current period."""
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 31, tzinfo=UTC)

    result = calculate_plan_change_proration(
        _proration_subscription(start=start, end=end),
        _plan(plan_id="current", amount=10000),
        _plan(plan_id="new", amount=20000),
        effective_date=datetime(2025, 12, 31, tzinfo=UTC),
    )

    assert isinstance(result, Failure)


def test_proration_requires_an_explicit_effective_date() -> None:
    """Proration must be deterministic and may not read the wall clock."""
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 31, tzinfo=UTC)

    result = calculate_plan_change_proration(
        _proration_subscription(start=start, end=end),
        _plan(plan_id="current", amount=10000),
        _plan(plan_id="new", amount=20000),
    )

    assert isinstance(result, Failure)


class _FakeWebhookRepository:
    def __init__(self, existing: object | None = None) -> None:
        self.existing = existing
        self.statuses: list[str] = []

    async def find_by_razorpay_event_id(self, _event_id: str):
        return Success(self.existing)

    async def find_by_id(self, _event_id: str):
        return Success(self.existing)

    async def create(self, event: object):
        if getattr(event, "retry_count", None) is None:
            event.retry_count = 0
        self.existing = event
        return Success(event)

    async def update_status(self, event: object, *, status: str, extra_values=None):
        self.statuses.append(status)
        event.status = status
        return Success(event)


def _webhook_service(
    *, webhooks: _FakeWebhookRepository, subscriptions: object, audit: object
) -> WebhookService:
    return WebhookService(
        webhooks=webhooks,  # type: ignore[arg-type]
        subscriptions=subscriptions,  # type: ignore[arg-type]
        plans=AsyncMock(),
        audit=audit,  # type: ignore[arg-type]
        payment_service=AsyncMock(),
        invoice_service=AsyncMock(),
    )


@pytest.mark.asyncio
async def test_webhook_duplicate_event_id_is_a_successful_noop() -> None:
    """Requirements 3.6-3.7 and 14.1-14.2: duplicate delivery is idempotent."""
    event = SimpleNamespace(status=WebhookEventStatus.PROCESSED.value)
    webhooks = _FakeWebhookRepository(existing=event)
    subscriptions = SimpleNamespace(find_by_razorpay_id=AsyncMock())
    service = _webhook_service(
        webhooks=webhooks,
        subscriptions=subscriptions,
        audit=AsyncMock(),
    )

    result = await service.process(
        event_id="evt_realistic_1",
        event_type=WebhookEventType.SUBSCRIPTION_ACTIVATED.value,
        payload={},
    )

    assert result == Success(True)
    subscriptions.find_by_razorpay_id.assert_not_awaited()
    assert webhooks.statuses == []


@pytest.mark.asyncio
async def test_failed_activation_replay_skips_already_active_subscription() -> None:
    """Requirement 31.3: replay does not reapply an already-active event."""
    event = SimpleNamespace(
        id="event-row-1",
        status=WebhookEventStatus.FAILED.value,
        event_type=WebhookEventType.SUBSCRIPTION_ACTIVATED.value,
        payload={"subscription": {"entity": {"id": "sub_remote_1"}}},
        retry_count=1,
    )
    webhooks = _FakeWebhookRepository(existing=event)
    subscriptions = SimpleNamespace(
        find_by_razorpay_id=AsyncMock(
            return_value=Success(_subscription(status=SubscriptionStatus.ACTIVE.value, version=9))
        )
    )
    audit = AsyncMock()
    service = _webhook_service(webhooks=webhooks, subscriptions=subscriptions, audit=audit)

    result = await service.replay("event-row-1")

    assert isinstance(result, Success)
    assert webhooks.statuses == [
        WebhookEventStatus.PROCESSING.value,
        WebhookEventStatus.SKIPPED.value,
    ]
    audit.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_webhook_audit_failure_is_reported_and_event_is_failed() -> None:
    """Current public behavior: audit failure makes webhook processing fail."""
    webhooks = _FakeWebhookRepository(existing=None)
    subscriptions = SimpleNamespace(
        find_by_razorpay_id=AsyncMock(
            return_value=Success(
                _subscription(status=SubscriptionStatus.AUTHENTICATED.value, version=2)
            )
        ),
        update_status=AsyncMock(
            return_value=Success(_subscription(status=SubscriptionStatus.ACTIVE.value, version=3))
        ),
    )
    audit = SimpleNamespace(
        create=AsyncMock(
            return_value=Failure(WebhookCollaboratorError(message="audit unavailable", details={}))
        )
    )
    service = _webhook_service(webhooks=webhooks, subscriptions=subscriptions, audit=audit)

    result = await service.process(
        event_id="evt_audit_failure_1",
        event_type=WebhookEventType.SUBSCRIPTION_ACTIVATED.value,
        payload={"subscription": {"entity": {"id": "sub_remote_1"}}},
    )

    assert isinstance(result, Failure)
    assert webhooks.statuses == [
        WebhookEventStatus.PROCESSING.value,
        WebhookEventStatus.FAILED.value,
    ]


class _PaymentRepository:
    def __init__(self) -> None:
        self.payment = SimpleNamespace(
            id="payment-1",
            subscription_id="subscription-1",
            razorpay_payment_id="pay_remote_1",
            amount=10000,
            currency="INR",
            status="captured",
            refund_amount=Decimal(0),
            metadata_={},
        )
        self.updated = False

    async def find_by_id_for_update(self, _payment_id: str):
        return Success(self.payment)

    async def update_status(self, *_args, **_kwargs):
        self.updated = True
        return Success(self.payment)


class _FailingRefundProvider:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def create_refund(self, **_kwargs: object):
        raise self.error


@pytest.mark.parametrize(
    ("provider_error", "expected_error_name"),
    [
        (
            RazorpayRetryableError(service="razorpay", detail="timeout"),
            "PaymentProviderUnavailableError",
        ),
        (
            RazorpayPermanentError(service="razorpay", detail="rejected"),
            "PaymentProviderError",
        ),
    ],
)
@pytest.mark.asyncio
async def test_payment_refund_preserves_provider_failure_classification(
    provider_error: Exception,
    expected_error_name: str,
) -> None:
    """Requirement 25: remote failure is typed and local state is not advanced."""
    payments = _PaymentRepository()
    service = PaymentService(
        payments=payments,  # type: ignore[arg-type]
        audit=AsyncMock(),
        razorpay=_FailingRefundProvider(provider_error),  # type: ignore[arg-type]
    )

    from app.features.billing.payments.dto import RefundRequestDTO

    result = await service.refund(
        "payment-1",
        RefundRequestDTO(amount=1000),
        user_id="user-1",
    )

    assert isinstance(result, Failure)
    assert type(result.failure()).__name__ == expected_error_name
    assert payments.updated is False
