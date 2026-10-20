"""Behavior and adapter-parity tests for the reconciliation provider seam."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest
from returns.result import Failure, Success

from app.features.billing.payments.clients.razorpay_client import (
    RazorpayPermanentError,
    RazorpayRetryableError,
)
from app.features.billing.providers import (
    CancelSubscriptionOperation,
    ChangeSubscriptionPlanOperation,
    CreateCustomerOperation,
    CreatePaymentLinkOperation,
    CreateRefundOperation,
    CreateSubscriptionOperation,
    FindCustomerByEmailOperation,
    FindPaymentLinkByCorrelationOperation,
    FindRefundByCorrelationOperation,
    FindSubscriptionByCorrelationOperation,
    InMemoryBillingProvider,
    ListPaymentsOperation,
    ObserveSubscriptionOperation,
    PauseSubscriptionOperation,
    ProviderAuthenticationError,
    ProviderErrorCode,
    ProviderOperationKind,
    ProviderOutcomeUnknownError,
    ProviderPaymentBatch,
    ProviderPaymentLink,
    ProviderPaymentState,
    ProviderRateLimitedError,
    ProviderReceipt,
    ProviderSubscriptionState,
    ProviderUnavailableError,
    RazorpayBillingProvider,
    ResumeSubscriptionOperation,
)

if TYPE_CHECKING:
    from typing import Any

    from app.features.billing.providers.protocol import ProviderResult


def _create_operation(correlation_id: str = "operation_1") -> CreateSubscriptionOperation:
    return CreateSubscriptionOperation(
        plan_id="plan_1",
        customer_id="customer_1",
        total_count=0,
        quantity=1,
        customer_notify=True,
        correlation_id=correlation_id,
    )


@pytest.mark.asyncio
async def test_in_memory_provider_is_stateful_by_correlation() -> None:
    provider = InMemoryBillingProvider()

    customer = await provider.execute(
        CreateCustomerOperation("customer@example.test", None, "Customer", "customer_1")
    )
    assert isinstance(customer, Success)

    found_customer = await provider.execute(
        FindCustomerByEmailOperation("customer@example.test", "find_customer_1")
    )
    assert isinstance(found_customer, Success)
    assert found_customer.unwrap().value == customer.unwrap().value

    subscription = await provider.execute(_create_operation())
    assert isinstance(subscription, Success)
    assert isinstance(subscription.unwrap().value, ProviderSubscriptionState)
    assert subscription.unwrap().value.grants_paid_entitlement is False

    found_subscription = await provider.execute(
        FindSubscriptionByCorrelationOperation("operation_1")
    )
    assert isinstance(found_subscription, Success)
    assert found_subscription.unwrap().value == subscription.unwrap().value

    link = await provider.execute(CreatePaymentLinkOperation(500, "INR", "upgrade", "link_1"))
    refund = await provider.execute(CreateRefundOperation("pay_1", 500, "refund_1"))
    assert isinstance(link, Success)
    assert isinstance(link.unwrap().value, ProviderPaymentLink)
    assert isinstance(refund, Success)
    assert isinstance(refund.unwrap().value, ProviderReceipt)
    assert isinstance(
        await provider.execute(FindPaymentLinkByCorrelationOperation("link_1")), Success
    )
    assert isinstance(
        await provider.execute(FindRefundByCorrelationOperation("pay_1", "refund_1")), Success
    )


@pytest.mark.asyncio
async def test_in_memory_provider_script_preserves_typed_unknown_failure() -> None:
    operation = _create_operation()
    scripted: tuple[ProviderResult, ...] = (
        Failure(ProviderOutcomeUnknownError(operation.kind, operation.correlation_id)),
    )
    provider = InMemoryBillingProvider({operation.kind: scripted})

    result = await provider.execute(operation)

    assert isinstance(result, Failure)
    assert isinstance(result.failure(), ProviderOutcomeUnknownError)
    assert result.failure().code is ProviderErrorCode.OUTCOME_UNKNOWN
    assert result.failure().requires_reconciliation is True


def test_operation_values_reject_invalid_mutation_inputs() -> None:
    with pytest.raises(ValueError, match="amount must be positive"):
        CreatePaymentLinkOperation(0, "INR", "upgrade", "link_1")
    with pytest.raises(ValueError, match="duration_days must be positive"):
        PauseSubscriptionOperation("sub_1", "pause_1", duration_days=0)
    with pytest.raises(ValueError, match="correlation_id must not be empty"):
        CreateRefundOperation("pay_1", 100, "")


class _FakeRazorpay:
    def __init__(self, *, subscription_payload: dict[str, Any] | None = None) -> None:
        self.subscription_payload = subscription_payload or {
            "id": "sub_remote_1",
            "status": "created",
            "current_start": 1_700_000_000,
            "current_end": 1_700_100_000,
            "short_url": "https://rzp.io/sub_remote_1",
        }
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def create_customer(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("create_customer", kwargs))
        return {"id": "cust_remote_1", "email": kwargs["email"]}

    async def find_customer_by_email(self, email: str) -> dict[str, Any] | None:
        self.calls.append(("find_customer_by_email", {"email": email}))
        return {"id": "cust_remote_1", "email": email}

    async def create_subscription(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("create_subscription", kwargs))
        return self.subscription_payload

    async def cancel_subscription(self, subscription_id: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("cancel_subscription", {"subscription_id": subscription_id, **kwargs}))
        return {"id": subscription_id, "status": "cancelled"}

    async def pause_subscription(self, subscription_id: str) -> dict[str, Any]:
        self.calls.append(("pause_subscription", {"subscription_id": subscription_id}))
        return {"id": subscription_id, "status": "paused"}

    async def resume_subscription(self, subscription_id: str) -> dict[str, Any]:
        self.calls.append(("resume_subscription", {"subscription_id": subscription_id}))
        return {"id": subscription_id, "status": "active"}

    async def change_subscription_plan(self, subscription_id: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(
            ("change_subscription_plan", {"subscription_id": subscription_id, **kwargs})
        )
        return {"id": subscription_id, "status": "active"}

    async def fetch_subscription(self, subscription_id: str) -> dict[str, Any]:
        self.calls.append(("fetch_subscription", {"subscription_id": subscription_id}))
        return {"id": subscription_id, "status": "active"}

    async def list_subscriptions(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(("list_subscriptions", kwargs))
        return [
            {
                "id": "sub_remote_1",
                "status": "created",
                "notes": {"billing_operation_id": "operation_1"},
            }
        ]

    async def create_payment_link(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("create_payment_link", kwargs))
        return {
            "id": "plink_remote_1",
            "short_url": "https://rzp.io/pay_remote_1",
            "status": "created",
        }

    async def list_payment_links(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(("list_payment_links", kwargs))
        return [
            {
                "id": "plink_remote_1",
                "short_url": "https://rzp.io/pay_remote_1",
                "status": "created",
                "notes": {"billing_operation_id": "link_1"},
            }
        ]

    async def create_refund(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("create_refund", kwargs))
        return {
            "id": "rfnd_remote_1",
            "payment_id": kwargs["payment_id"],
            "amount": kwargs["amount"],
        }

    async def list_refunds(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(("list_refunds", kwargs))
        return [
            {
                "id": "rfnd_remote_1",
                "payment_id": kwargs["payment_id"],
                "amount": 500,
                "notes": {"billing_operation_id": "refund_1"},
            }
        ]


@pytest.mark.asyncio
async def test_razorpay_adapter_normalizes_subscription_and_carries_correlation() -> None:
    client = _FakeRazorpay()
    provider = RazorpayBillingProvider(client)

    result = await provider.execute(_create_operation())

    assert isinstance(result, Success)
    assert result.unwrap().operation is ProviderOperationKind.CREATE_SUBSCRIPTION
    assert isinstance(result.unwrap().value, ProviderSubscriptionState)
    assert result.unwrap().value.grants_paid_entitlement is False
    assert client.calls[0][1]["notes"]["billing_operation_id"] == "operation_1"


@pytest.mark.asyncio
async def test_razorpay_adapter_uses_documented_pause_resume_post_shapes() -> None:
    client = _FakeRazorpay()
    provider = RazorpayBillingProvider(client)

    pause = await provider.execute(PauseSubscriptionOperation("sub_remote_1", "pause_1", 30))
    resume = await provider.execute(ResumeSubscriptionOperation("sub_remote_1", "resume_1"))
    change = await provider.execute(
        ChangeSubscriptionPlanOperation("sub_remote_1", "plan_2", "change_1", "cycle_end")
    )

    assert all(isinstance(result, Success) for result in (pause, resume, change))
    assert [call[0] for call in client.calls] == [
        "pause_subscription",
        "resume_subscription",
        "change_subscription_plan",
    ]


@pytest.mark.asyncio
async def test_razorpay_adapter_supports_correlation_reconciliation_reads() -> None:
    provider = RazorpayBillingProvider(_FakeRazorpay())

    subscription = await provider.execute(FindSubscriptionByCorrelationOperation("operation_1"))
    link = await provider.execute(FindPaymentLinkByCorrelationOperation("link_1"))
    refund = await provider.execute(FindRefundByCorrelationOperation("pay_1", "refund_1"))

    assert all(isinstance(result, Success) for result in (subscription, link, refund))
    assert isinstance(subscription.unwrap().value, ProviderSubscriptionState)
    assert isinstance(link.unwrap().value, ProviderPaymentLink)
    assert isinstance(refund.unwrap().value, ProviderReceipt)


@pytest.mark.asyncio
async def test_payment_list_observation_is_typed_and_read_only() -> None:
    class PaymentRazorpay(_FakeRazorpay):
        async def list_payments(self, *, params: dict[str, Any]) -> list[dict[str, Any]]:
            self.calls.append(("list_payments", params))
            return [
                {
                    "id": "pay_remote_1",
                    "status": "captured",
                    "amount": 500,
                    "currency": "INR",
                    "subscription_id": "sub_remote_1",
                    "method": "upi",
                    "created_at": 1_700_000_000,
                    "captured_at": 1_700_000_001,
                }
            ]

    operation = ListPaymentsOperation(
        correlation_id="reconciliation_1",
        from_timestamp=1_699_000_000,
        to_timestamp=1_700_100_000,
    )
    client = PaymentRazorpay()
    result = await RazorpayBillingProvider(client).execute(operation)

    assert isinstance(result, Success)
    assert result.unwrap().operation is operation.kind
    assert isinstance(result.unwrap().value, ProviderPaymentBatch)
    assert result.unwrap().value.payments == (
        ProviderPaymentState(
            provider_payment_id="pay_remote_1",
            status="captured",
            amount=500,
            currency="INR",
            subscription_id="sub_remote_1",
            method="upi",
            created_at=datetime.fromtimestamp(1_700_000_000, tz=UTC),
            captured_at=datetime.fromtimestamp(1_700_000_001, tz=UTC),
        ),
    )
    assert client.calls == [
        (
            "list_payments",
            {"count": 100, "from": 1_699_000_000, "to": 1_700_100_000},
        )
    ]


@pytest.mark.asyncio
async def test_in_memory_payment_observation_preserves_typed_facts() -> None:
    payment = ProviderPaymentState(
        provider_payment_id="pay_memory_1",
        status="captured",
        amount=500,
        currency="INR",
    )
    result = await InMemoryBillingProvider(payments=(payment,)).execute(
        ListPaymentsOperation(correlation_id="memory_reconciliation")
    )

    assert isinstance(result, Success)
    assert isinstance(result.unwrap().value, ProviderPaymentBatch)
    assert result.unwrap().value.payments == (payment,)


@pytest.mark.asyncio
async def test_razorpay_retryable_mutation_is_unknown_but_reads_are_typed() -> None:
    class BrokenRazorpay(_FakeRazorpay):
        async def create_subscription(self, **kwargs: Any) -> dict[str, Any]:
            del kwargs
            raise RazorpayRetryableError(service="razorpay", detail="transport failure")

        async def fetch_subscription(self, subscription_id: str) -> dict[str, Any]:
            del subscription_id
            raise RazorpayRetryableError(service="razorpay", detail="transport failure")

    provider = RazorpayBillingProvider(BrokenRazorpay())

    mutation = await provider.execute(_create_operation())
    observation = await provider.execute(ObserveSubscriptionOperation("sub_remote_1", "observe_1"))

    assert isinstance(mutation, Failure)
    assert isinstance(mutation.failure(), ProviderOutcomeUnknownError)
    assert isinstance(observation, Failure)
    assert isinstance(observation.failure(), ProviderUnavailableError)


@pytest.mark.asyncio
async def test_razorpay_429_read_is_rate_limited() -> None:
    class RateLimitedRazorpay(_FakeRazorpay):
        async def fetch_subscription(self, subscription_id: str) -> dict[str, Any]:
            del subscription_id
            raise RazorpayRetryableError(service="razorpay", detail="rate limited", status_code=429)

    result = await RazorpayBillingProvider(RateLimitedRazorpay()).execute(
        ObserveSubscriptionOperation("sub_remote_1", "observe_429")
    )

    assert isinstance(result, Failure)
    assert isinstance(result.failure(), ProviderRateLimitedError)


@pytest.mark.asyncio
async def test_razorpay_403_is_authentication_not_substring_parsed() -> None:
    class RejectedRazorpay(_FakeRazorpay):
        async def cancel_subscription(self, subscription_id: str, **kwargs: Any) -> dict[str, Any]:
            del subscription_id, kwargs
            raise RazorpayPermanentError(
                service="razorpay", detail="secret details", status_code=403
            )

    result = await RazorpayBillingProvider(RejectedRazorpay()).execute(
        CancelSubscriptionOperation("sub_remote_1", False, "cancel_1")
    )

    assert isinstance(result, Failure)
    assert isinstance(result.failure(), ProviderAuthenticationError)
    assert result.failure().code is ProviderErrorCode.AUTHENTICATION
    assert "secret" not in result.failure().message


@pytest.mark.asyncio
async def test_malformed_mutation_response_requires_reconciliation() -> None:
    client = _FakeRazorpay(subscription_payload={"id": "sub_remote_1"})
    result = await RazorpayBillingProvider(client).execute(_create_operation())

    assert isinstance(result, Failure)
    assert isinstance(result.failure(), ProviderOutcomeUnknownError)


@pytest.mark.asyncio
async def test_missing_correlation_match_requires_manual_review() -> None:
    class NoMatchRazorpay(_FakeRazorpay):
        async def list_payment_links(self, **kwargs: Any) -> list[dict[str, Any]]:
            del kwargs
            return []

    result = await RazorpayBillingProvider(NoMatchRazorpay()).execute(
        FindPaymentLinkByCorrelationOperation("not_found")
    )

    assert isinstance(result, Failure)
    assert result.failure().code is ProviderErrorCode.MANUAL_REVIEW_REQUIRED


@pytest.mark.asyncio
async def test_provider_client_mutations_are_not_automatically_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.features.billing.payments.clients.razorpay_client import RazorpayClient

    client = RazorpayClient(key_id="key", key_secret="secret", base_url="https://example.test")
    request = AsyncMock(side_effect=RazorpayRetryableError(service="razorpay", detail="timeout"))
    monkeypatch.setattr(client, "_request", request)

    with pytest.raises(RazorpayRetryableError):
        await client.create_refund(payment_id="pay_1", amount=100)

    request.assert_awaited_once()


@pytest.mark.asyncio
async def test_provider_client_pause_and_resume_request_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.features.billing.payments.clients.razorpay_client import RazorpayClient

    class RecordingHttpClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, dict[str, Any] | None]] = []

        async def __aenter__(self) -> RecordingHttpClient:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def request(
            self,
            method: str,
            path: str,
            *,
            params: dict[str, Any] | None = None,
            json: dict[str, Any] | None = None,
        ) -> Any:
            self.calls.append((method, path, json))

            class Response:
                status_code = 200

                @staticmethod
                def json() -> dict[str, Any]:
                    return {"id": "sub_1", "status": "active"}

            return Response()

    recording = RecordingHttpClient()
    client = RazorpayClient(key_id="key", key_secret="secret", base_url="https://example.test")
    monkeypatch.setattr(client, "_client", lambda: recording)

    await client.pause_subscription("sub_1")
    await client.resume_subscription("sub_1")

    assert recording.calls == [
        ("POST", "/v1/subscriptions/sub_1/pause", {"pause_at": "now"}),
        ("POST", "/v1/subscriptions/sub_1/resume", {"resume_at": "now"}),
    ]
