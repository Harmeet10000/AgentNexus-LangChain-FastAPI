"""Razorpay adapter for the provider-neutral billing port."""

from __future__ import annotations

from datetime import UTC, datetime
from inspect import Parameter, signature
from typing import TYPE_CHECKING, Protocol, assert_never, override

from returns.result import Failure, Success

from app.features.billing.payments.clients.razorpay_client import (
    CircuitOpenError,
    RazorpayClient,
    RazorpayPermanentError,
    RazorpayRetryableError,
)

from .protocol import (
    BillingProvider,
    CancelSubscriptionOperation,
    ChangeSubscriptionPlanOperation,
    CreateCustomerOperation,
    CreatePaymentLinkOperation,
    CreatePlanOperation,
    CreateRefundOperation,
    CreateSubscriptionOperation,
    FindCustomerByEmailOperation,
    FindPaymentLinkByCorrelationOperation,
    FindPlanByCorrelationOperation,
    FindRefundByCorrelationOperation,
    FindSubscriptionByCorrelationOperation,
    ListPaymentsOperation,
    ObserveSubscriptionOperation,
    PauseSubscriptionOperation,
    ProviderAuthenticationError,
    ProviderCustomer,
    ProviderInvalidRequestError,
    ProviderManualReviewRequiredError,
    ProviderNotFoundError,
    ProviderOperationKind,
    ProviderOutcomeUnknownError,
    ProviderPaymentBatch,
    ProviderPaymentLink,
    ProviderPaymentState,
    ProviderPlan,
    ProviderRateLimitedError,
    ProviderReceipt,
    ProviderReconciliationRequiredError,
    ProviderSubscriptionState,
    ProviderSuccess,
    ProviderUnavailableError,
    ProviderUnexpectedError,
    ResumeSubscriptionOperation,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from typing import Any, Literal

    from .protocol import ProviderError, ProviderOperation, ProviderResult, ProviderValue


class _RazorpayClientPort(Protocol):
    async def create_customer(
        self,
        *,
        email: str,
        contact: str | None,
        name: str | None,
        notes: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    async def find_customer_by_email(self, email: str) -> dict[str, Any] | None: ...

    async def create_plan(
        self,
        *,
        name: str,
        amount: int,
        interval: str,
        currency: str,
        period: int,
        notes: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    async def list_plans(self, *, params: dict[str, Any]) -> list[dict[str, Any]]: ...

    async def create_subscription(
        self,
        *,
        plan_id: str,
        customer_id: str,
        total_count: int,
        quantity: int,
        customer_notify: bool,
        notes: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    async def cancel_subscription(
        self, subscription_id: str, *, cancel_at_cycle_end: bool = False
    ) -> dict[str, Any]: ...

    async def pause_subscription(
        self, subscription_id: str, *, pause_at: Literal["now"] = "now"
    ) -> dict[str, Any]: ...

    async def resume_subscription(
        self, subscription_id: str, *, resume_at: Literal["now"] = "now"
    ) -> dict[str, Any]: ...

    async def change_subscription_plan(
        self,
        subscription_id: str,
        *,
        plan_id: str,
        schedule_at: Literal["now", "cycle_end"] = "now",
    ) -> dict[str, Any]: ...

    async def fetch_subscription(self, subscription_id: str) -> dict[str, Any]: ...

    async def list_subscriptions(self, *, params: dict[str, Any]) -> list[dict[str, Any]]: ...

    async def list_payments(self, *, params: dict[str, Any]) -> list[dict[str, Any]]: ...

    async def create_payment_link(
        self,
        *,
        amount: int,
        currency: str,
        description: str,
        customer_id: str | None = None,
        notes: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    async def list_payment_links(self, *, params: dict[str, Any]) -> list[dict[str, Any]]: ...

    async def create_refund(
        self, *, payment_id: str, amount: int, notes: dict[str, Any] | None = None
    ) -> dict[str, Any]: ...

    async def list_refunds(
        self, *, payment_id: str, params: dict[str, Any]
    ) -> list[dict[str, Any]]: ...


_MUTATING_KINDS = frozenset(
    {
        ProviderOperationKind.CREATE_CUSTOMER,
        ProviderOperationKind.CREATE_SUBSCRIPTION,
        ProviderOperationKind.CANCEL_SUBSCRIPTION,
        ProviderOperationKind.PAUSE_SUBSCRIPTION,
        ProviderOperationKind.RESUME_SUBSCRIPTION,
        ProviderOperationKind.CHANGE_SUBSCRIPTION_PLAN,
        ProviderOperationKind.CREATE_PAYMENT_LINK,
        ProviderOperationKind.CREATE_REFUND,
        ProviderOperationKind.CREATE_PLAN,
    }
)

_SUBSCRIPTION_STATUSES = frozenset(
    {
        "created",
        "authenticated",
        "active",
        "pending",
        "paused",
        "halted",
        "cancelled",
        "completed",
        "expired",
    }
)
_CREATE_OR_UPDATE_STATUSES = frozenset({"created", "authenticated", "active", "pending"})
_PAYMENT_LINK_STATUSES = frozenset(
    {"created", "partially_paid", "expired", "cancelled", "paid", "issued"}
)


def _notes(values: Mapping[str, str], correlation_id: str) -> dict[str, Any]:
    result = dict(values)
    result.setdefault("billing_operation_id", correlation_id)
    return result


def _accepts_keyword(method: Callable[..., object], name: str) -> bool:
    """Check optional adapter capability without issuing a second mutation."""

    try:
        method_signature = signature(method)
    except (TypeError, ValueError):
        return False
    return name in method_signature.parameters or any(
        parameter.kind is Parameter.VAR_KEYWORD
        for parameter in method_signature.parameters.values()
    )


def _epoch(value: object) -> datetime | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return datetime.fromtimestamp(value, tz=UTC)


def _error_for_permanent(
    exc: RazorpayPermanentError, operation: ProviderOperationKind, correlation_id: str
) -> ProviderError:
    status_code = getattr(exc, "razorpay_status_code", None)
    if status_code in {401, 403}:
        return ProviderAuthenticationError(operation=operation, correlation_id=correlation_id)
    if status_code == 404:
        return ProviderNotFoundError(operation=operation, correlation_id=correlation_id)
    return ProviderInvalidRequestError(operation=operation, correlation_id=correlation_id)


def _subscription_state(
    payload: dict[str, Any],
    operation: ProviderOperationKind,
    correlation_id: str,
    *,
    expected_statuses: frozenset[str] | None = None,
) -> ProviderResult[ProviderValue]:
    if not isinstance(payload, dict):
        return Failure(
            ProviderOutcomeUnknownError(operation, correlation_id)
            if operation in _MUTATING_KINDS
            else ProviderReconciliationRequiredError(
                operation=operation, correlation_id=correlation_id
            )
        )
    provider_id = payload.get("id")
    status = payload.get("status")
    plan_id = payload.get("plan_id")
    invalid_identity = not isinstance(provider_id, str) or not provider_id
    invalid_status = not isinstance(status, str) or status not in _SUBSCRIPTION_STATUSES
    invalid_plan_id = plan_id is not None and (not isinstance(plan_id, str) or not plan_id.strip())
    unexpected_status = expected_statuses is not None and status not in expected_statuses
    if invalid_identity or invalid_status or invalid_plan_id or unexpected_status:
        return Failure(
            ProviderOutcomeUnknownError(
                operation=operation,
                correlation_id=correlation_id,
                message="Provider subscription response requires reconciliation",
            )
            if operation in _MUTATING_KINDS
            else ProviderReconciliationRequiredError(
                operation=operation, correlation_id=correlation_id
            )
        )
    cancel_at_cycle_end = payload.get("cancel_at_cycle_end")
    return Success(
        ProviderSuccess(
            operation=operation,
            correlation_id=correlation_id,
            value=ProviderSubscriptionState(
                provider_subscription_id=provider_id,
                status=status,
                current_period_start=_epoch(payload.get("current_start")),
                current_period_end=_epoch(payload.get("current_end")),
                ended_at=_epoch(payload.get("ended_at")),
                plan_id=plan_id if isinstance(plan_id, str) else None,
                payment_url=(
                    payload.get("short_url") if isinstance(payload.get("short_url"), str) else None
                ),
                cancel_at_cycle_end=(
                    cancel_at_cycle_end if isinstance(cancel_at_cycle_end, bool) else None
                ),
            ),
        )
    )


def _payment_state(
    payload: object,
) -> ProviderPaymentState | None:
    if not isinstance(payload, dict):
        return None
    provider_id = payload.get("id")
    status = payload.get("status")
    amount = payload.get("amount")
    currency = payload.get("currency")
    invalid_identity = not isinstance(provider_id, str) or not provider_id
    invalid_status = not isinstance(status, str) or not status
    invalid_amount = isinstance(amount, bool) or not isinstance(amount, int) or amount <= 0
    invalid_currency = not isinstance(currency, str) or not currency
    if invalid_identity or invalid_status or invalid_amount or invalid_currency:
        return None
    subscription_id = payload.get("subscription_id")
    method = payload.get("method")
    return ProviderPaymentState(
        provider_payment_id=provider_id,
        status=status,
        amount=amount,
        currency=currency,
        subscription_id=subscription_id if isinstance(subscription_id, str) else None,
        method=method if isinstance(method, str) else None,
        created_at=_epoch(payload.get("created_at")),
        captured_at=_epoch(payload.get("captured_at")),
    )


class RazorpayBillingProvider(BillingProvider):
    """Translate provider-neutral operations to the Razorpay client."""

    def __init__(self, client: _RazorpayClientPort | None = None) -> None:
        self._client = client or RazorpayClient()

    @override
    async def execute(self, operation: ProviderOperation) -> ProviderResult[ProviderValue]:
        kind = operation.kind
        correlation_id = operation.correlation_id
        try:
            return await self._execute(operation)
        except CircuitOpenError as exc:
            exc.add_note(f"operation={kind}, correlation_id={correlation_id}")
            return Failure(ProviderUnavailableError(operation=kind, correlation_id=correlation_id))
        except RazorpayRetryableError as exc:
            exc.add_note(f"operation={kind}, correlation_id={correlation_id}")
            if kind in _MUTATING_KINDS:
                return Failure(
                    ProviderOutcomeUnknownError(operation=kind, correlation_id=correlation_id)
                )
            if getattr(exc, "razorpay_status_code", None) == 429:
                return Failure(
                    ProviderRateLimitedError(operation=kind, correlation_id=correlation_id)
                )
            return Failure(ProviderUnavailableError(operation=kind, correlation_id=correlation_id))
        except RazorpayPermanentError as exc:
            exc.add_note(f"operation={kind}, correlation_id={correlation_id}")
            return Failure(_error_for_permanent(exc, kind, correlation_id))
        except Exception as exc:  # noqa: BLE001 — third-party boundary sanitization
            exc.add_note(f"operation={kind}, correlation_id={correlation_id}")
            if kind in _MUTATING_KINDS:
                return Failure(
                    ProviderOutcomeUnknownError(operation=kind, correlation_id=correlation_id)
                )
            return Failure(ProviderUnexpectedError(operation=kind, correlation_id=correlation_id))

    async def _execute(self, operation: ProviderOperation) -> ProviderResult[ProviderValue]:  # noqa: PLR0912, PLR0915, PLR0914 — closed provider union dispatch
        if isinstance(operation, CreateCustomerOperation):
            payload = await self._client.create_customer(
                email=operation.email,
                contact=operation.contact,
                name=operation.name,
                notes=_notes(operation.notes, operation.correlation_id),
            )
            provider_id = payload.get("id")
            email = payload.get("email")
            if (
                not isinstance(provider_id, str)
                or not provider_id
                or not isinstance(email, str)
                or email.lower() != operation.email.lower()
            ):
                return Failure(
                    ProviderOutcomeUnknownError(
                        operation=operation.kind, correlation_id=operation.correlation_id
                    )
                )
            return Success(
                ProviderSuccess(
                    operation=operation.kind,
                    correlation_id=operation.correlation_id,
                    value=ProviderCustomer(provider_customer_id=provider_id, email=email),
                )
            )

        if isinstance(operation, FindCustomerByEmailOperation):
            payload = await self._client.find_customer_by_email(operation.email)
            if payload is None:
                return Failure(
                    ProviderNotFoundError(
                        operation=operation.kind, correlation_id=operation.correlation_id
                    )
                )
            provider_id = payload.get("id")
            email = payload.get("email")
            if (
                not isinstance(provider_id, str)
                or not provider_id
                or not isinstance(email, str)
                or email.lower() != operation.email.lower()
            ):
                return Failure(
                    ProviderReconciliationRequiredError(
                        operation=operation.kind, correlation_id=operation.correlation_id
                    )
                )
            return Success(
                ProviderSuccess(
                    operation=operation.kind,
                    correlation_id=operation.correlation_id,
                    value=ProviderCustomer(provider_customer_id=provider_id, email=email),
                )
            )

        if isinstance(operation, CreateSubscriptionOperation):
            payload = await self._client.create_subscription(
                plan_id=operation.plan_id,
                customer_id=operation.customer_id,
                total_count=operation.total_count,
                quantity=operation.quantity,
                customer_notify=operation.customer_notify,
                notes=_notes(operation.notes, operation.correlation_id),
            )
            return _subscription_state(
                payload,
                operation.kind,
                operation.correlation_id,
                expected_statuses=_CREATE_OR_UPDATE_STATUSES,
            )

        if isinstance(operation, CancelSubscriptionOperation):
            payload = await self._client.cancel_subscription(
                operation.provider_subscription_id,
                cancel_at_cycle_end=operation.cancel_at_cycle_end,
            )
            return _subscription_state(
                payload,
                operation.kind,
                operation.correlation_id,
                expected_statuses=frozenset({"cancelled", "active"}),
            )

        if isinstance(operation, PauseSubscriptionOperation):
            pause = self._client.pause_subscription
            payload = (
                await pause(
                    operation.provider_subscription_id,
                    pause_at=operation.pause_at,
                )
                if _accepts_keyword(pause, "pause_at")
                else await pause(operation.provider_subscription_id)
            )
            return _subscription_state(
                payload,
                operation.kind,
                operation.correlation_id,
                expected_statuses=frozenset({"paused", "cancelled"}),
            )

        if isinstance(operation, ResumeSubscriptionOperation):
            resume = self._client.resume_subscription
            payload = (
                await resume(
                    operation.provider_subscription_id,
                    resume_at=operation.resume_at,
                )
                if _accepts_keyword(resume, "resume_at")
                else await resume(operation.provider_subscription_id)
            )
            return _subscription_state(
                payload,
                operation.kind,
                operation.correlation_id,
                expected_statuses=frozenset({"active"}),
            )

        if isinstance(operation, ChangeSubscriptionPlanOperation):
            payload = await self._client.change_subscription_plan(
                operation.provider_subscription_id,
                plan_id=operation.plan_id,
                schedule_at=operation.schedule_at,
            )
            return _subscription_state(
                payload,
                operation.kind,
                operation.correlation_id,
                expected_statuses=_CREATE_OR_UPDATE_STATUSES,
            )

        if isinstance(operation, ObserveSubscriptionOperation):
            payload = await self._client.fetch_subscription(operation.provider_subscription_id)
            return _subscription_state(payload, operation.kind, operation.correlation_id)

        if isinstance(operation, FindSubscriptionByCorrelationOperation):
            payloads = await self._client.list_subscriptions(params={"count": 100})
            matches = [
                payload
                for payload in payloads
                if isinstance(payload.get("notes"), dict)
                and payload["notes"].get("billing_operation_id") == operation.correlation_id
            ]
            if len(matches) != 1:
                return Failure(
                    ProviderManualReviewRequiredError(
                        operation=operation.kind, correlation_id=operation.correlation_id
                    )
                )
            return _subscription_state(matches[0], operation.kind, operation.correlation_id)

        if isinstance(operation, ListPaymentsOperation):
            params: dict[str, Any] = {"count": operation.count}
            if operation.from_timestamp is not None:
                params["from"] = operation.from_timestamp
            if operation.to_timestamp is not None:
                params["to"] = operation.to_timestamp
            payloads = await self._client.list_payments(params=params)
            payments: list[ProviderPaymentState] = []
            for payload in payloads:
                state = _payment_state(payload)
                if state is None:
                    return Failure(
                        ProviderReconciliationRequiredError(
                            operation=operation.kind,
                            correlation_id=operation.correlation_id,
                        )
                    )
                payments.append(state)
            return Success(
                ProviderSuccess(
                    operation=operation.kind,
                    correlation_id=operation.correlation_id,
                    value=ProviderPaymentBatch(payments=tuple(payments)),
                )
            )

        if isinstance(operation, CreatePaymentLinkOperation):
            payload = await self._client.create_payment_link(
                amount=operation.amount,
                currency=operation.currency,
                description=operation.description,
                customer_id=operation.customer_id,
                notes=_notes(operation.notes, operation.correlation_id),
            )
            provider_id = payload.get("id")
            if not isinstance(provider_id, str) or not provider_id:
                return Failure(
                    ProviderOutcomeUnknownError(
                        operation=operation.kind, correlation_id=operation.correlation_id
                    )
                )
            status = payload.get("status")
            if not isinstance(status, str) or status not in _PAYMENT_LINK_STATUSES:
                return Failure(
                    ProviderOutcomeUnknownError(
                        operation=operation.kind, correlation_id=operation.correlation_id
                    )
                )
            return Success(
                ProviderSuccess(
                    operation=operation.kind,
                    correlation_id=operation.correlation_id,
                    value=ProviderPaymentLink(
                        provider_payment_link_id=provider_id,
                        short_url=payload.get("short_url")
                        if isinstance(payload.get("short_url"), str)
                        else None,
                        status=status,
                    ),
                )
            )

        if isinstance(operation, CreatePlanOperation):
            create_plan = self._client.create_plan
            if _accepts_keyword(create_plan, "notes"):
                payload = await create_plan(
                    name=operation.name,
                    amount=operation.amount,
                    interval=operation.period,
                    currency=operation.currency,
                    period=operation.interval,
                    notes=_notes(operation.notes, operation.correlation_id),
                )
            else:
                payload = await create_plan(
                    name=operation.name,
                    amount=operation.amount,
                    interval=operation.period,
                    currency=operation.currency,
                    period=operation.interval,
                )
            provider_id = payload.get("id")
            item = payload.get("item")
            name = item.get("name") if isinstance(item, dict) else None
            if not isinstance(provider_id, str) or not provider_id:
                return Failure(
                    ProviderOutcomeUnknownError(
                        operation=operation.kind, correlation_id=operation.correlation_id
                    )
                )
            return Success(
                ProviderSuccess(
                    operation=operation.kind,
                    correlation_id=operation.correlation_id,
                    value=ProviderPlan(
                        provider_plan_id=provider_id,
                        name=name if isinstance(name, str) else operation.name,
                    ),
                )
            )

        if isinstance(operation, FindPlanByCorrelationOperation):
            payloads = await self._client.list_plans(params={"count": 100})
            matches = [
                payload
                for payload in payloads
                if isinstance(payload.get("notes"), dict)
                and payload["notes"].get("billing_operation_id") == operation.correlation_id
            ]
            if len(matches) != 1:
                return Failure(
                    ProviderManualReviewRequiredError(
                        operation=operation.kind, correlation_id=operation.correlation_id
                    )
                )
            provider_id = matches[0].get("id")
            if not isinstance(provider_id, str) or not provider_id:
                return Failure(
                    ProviderReconciliationRequiredError(
                        operation=operation.kind, correlation_id=operation.correlation_id
                    )
                )
            item = matches[0].get("item")
            name = item.get("name") if isinstance(item, dict) else None
            return Success(
                ProviderSuccess(
                    operation=operation.kind,
                    correlation_id=operation.correlation_id,
                    value=ProviderPlan(
                        provider_plan_id=provider_id,
                        name=name if isinstance(name, str) else None,
                    ),
                )
            )

        if isinstance(operation, FindPaymentLinkByCorrelationOperation):
            payloads = await self._client.list_payment_links(params={"count": 100})
            matches = [
                payload
                for payload in payloads
                if isinstance(payload.get("notes"), dict)
                and payload["notes"].get("billing_operation_id") == operation.correlation_id
            ]
            if len(matches) != 1:
                return Failure(
                    ProviderManualReviewRequiredError(
                        operation=operation.kind, correlation_id=operation.correlation_id
                    )
                )
            provider_id = matches[0].get("id")
            status = matches[0].get("status")
            if not isinstance(provider_id, str) or not provider_id:
                return Failure(
                    ProviderReconciliationRequiredError(
                        operation=operation.kind, correlation_id=operation.correlation_id
                    )
                )
            if not isinstance(status, str) or status not in _PAYMENT_LINK_STATUSES:
                return Failure(
                    ProviderReconciliationRequiredError(
                        operation=operation.kind, correlation_id=operation.correlation_id
                    )
                )
            return Success(
                ProviderSuccess(
                    operation=operation.kind,
                    correlation_id=operation.correlation_id,
                    value=ProviderPaymentLink(
                        provider_payment_link_id=provider_id,
                        short_url=matches[0].get("short_url")
                        if isinstance(matches[0].get("short_url"), str)
                        else None,
                        status=status,
                    ),
                )
            )

        if isinstance(operation, CreateRefundOperation):
            payload = await self._client.create_refund(
                payment_id=operation.payment_id,
                amount=operation.amount,
                notes=_notes(operation.notes, operation.correlation_id),
            )
            provider_id = payload.get("id")
            payment_id = payload.get("payment_id")
            amount = payload.get("amount")
            if (
                not isinstance(provider_id, str)
                or not provider_id
                or payment_id != operation.payment_id
                or amount != operation.amount
            ):
                return Failure(
                    ProviderOutcomeUnknownError(
                        operation=operation.kind, correlation_id=operation.correlation_id
                    )
                )
            return Success(
                ProviderSuccess(
                    operation=operation.kind,
                    correlation_id=operation.correlation_id,
                    value=ProviderReceipt(
                        provider_refund_id=provider_id,
                        payment_id=operation.payment_id,
                        amount=operation.amount,
                    ),
                )
            )

        if isinstance(operation, FindRefundByCorrelationOperation):
            payloads = await self._client.list_refunds(
                payment_id=operation.payment_id, params={"count": 100}
            )
            matches = [
                payload
                for payload in payloads
                if isinstance(payload.get("notes"), dict)
                and payload["notes"].get("billing_operation_id") == operation.correlation_id
            ]
            if len(matches) != 1:
                return Failure(
                    ProviderManualReviewRequiredError(
                        operation=operation.kind, correlation_id=operation.correlation_id
                    )
                )
            provider_id = matches[0].get("id")
            amount = matches[0].get("amount")
            invalid_provider_id = not isinstance(provider_id, str) or not provider_id
            wrong_payment = matches[0].get("payment_id") != operation.payment_id
            invalid_amount = isinstance(amount, bool) or not isinstance(amount, int) or amount <= 0
            if invalid_provider_id or wrong_payment or invalid_amount:
                return Failure(
                    ProviderReconciliationRequiredError(
                        operation=operation.kind, correlation_id=operation.correlation_id
                    )
                )
            return Success(
                ProviderSuccess(
                    operation=operation.kind,
                    correlation_id=operation.correlation_id,
                    value=ProviderReceipt(
                        provider_refund_id=provider_id,
                        payment_id=operation.payment_id,
                        amount=amount,
                    ),
                )
            )

        assert_never(operation)
