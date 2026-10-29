"""Stateful in-memory billing provider for coordinator/reconciliation tests."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, assert_never, override

from returns.result import Failure, Success

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
    ProviderCustomer,
    ProviderManualReviewRequiredError,
    ProviderNotFoundError,
    ProviderOperationKind,
    ProviderPaymentBatch,
    ProviderPaymentLink,
    ProviderPlan,
    ProviderReceipt,
    ProviderSubscriptionState,
    ProviderSuccess,
    ResumeSubscriptionOperation,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .protocol import (
        ProviderOperation,
        ProviderPaymentState,
        ProviderResult,
        ProviderValue,
    )


@dataclass(frozen=True, slots=True)
class InMemoryCall:
    """Recorded operation supplied to the provider port."""

    operation: ProviderOperation


class InMemoryBillingProvider(BillingProvider):
    """Stateful provider double with optional FIFO scripted failures."""

    def __init__(
        self,
        script: Mapping[ProviderOperationKind, tuple[ProviderResult[ProviderValue], ...]]
        | None = None,
        *,
        payments: tuple[ProviderPaymentState, ...] = (),
    ) -> None:
        self._script = {kind: deque(outcomes) for kind, outcomes in (script or {}).items()}
        self._calls: list[InMemoryCall] = []
        self._customers_by_email: dict[str, str] = {}
        self._customers_by_correlation: dict[str, ProviderCustomer] = {}
        self._subscriptions_by_correlation: dict[str, ProviderSubscriptionState] = {}
        self._subscriptions_by_id: dict[str, ProviderSubscriptionState] = {}
        self._payment_links_by_correlation: dict[str, ProviderPaymentLink] = {}
        self._refunds_by_correlation: dict[str, ProviderReceipt] = {}
        self._plans_by_correlation: dict[str, ProviderPlan] = {}
        self._completed_mutations: dict[str, ProviderSuccess[ProviderValue]] = {}
        self._payments = payments

    @property
    def calls(self) -> tuple[InMemoryCall, ...]:
        return tuple(self._calls)

    @override
    async def execute(self, operation: ProviderOperation) -> ProviderResult[ProviderValue]:
        self._calls.append(InMemoryCall(operation))
        scripted = self._script.get(operation.kind)
        if scripted:
            return scripted.popleft()
        if operation.kind in {
            ProviderOperationKind.CREATE_CUSTOMER,
            ProviderOperationKind.CREATE_SUBSCRIPTION,
            ProviderOperationKind.CANCEL_SUBSCRIPTION,
            ProviderOperationKind.PAUSE_SUBSCRIPTION,
            ProviderOperationKind.RESUME_SUBSCRIPTION,
            ProviderOperationKind.CHANGE_SUBSCRIPTION_PLAN,
            ProviderOperationKind.CREATE_PAYMENT_LINK,
            ProviderOperationKind.CREATE_REFUND,
            ProviderOperationKind.CREATE_PLAN,
        }:
            completed = self._completed_mutations.get(operation.correlation_id)
            if completed is not None:
                return Success(completed)
        result = self._default_result(operation)
        if isinstance(result, Success) and operation.kind in {
            ProviderOperationKind.CREATE_CUSTOMER,
            ProviderOperationKind.CREATE_SUBSCRIPTION,
            ProviderOperationKind.CANCEL_SUBSCRIPTION,
            ProviderOperationKind.PAUSE_SUBSCRIPTION,
            ProviderOperationKind.RESUME_SUBSCRIPTION,
            ProviderOperationKind.CHANGE_SUBSCRIPTION_PLAN,
            ProviderOperationKind.CREATE_PAYMENT_LINK,
            ProviderOperationKind.CREATE_REFUND,
            ProviderOperationKind.CREATE_PLAN,
        }:
            self._completed_mutations[operation.correlation_id] = result.unwrap()
        return result

    def _default_result(self, operation: ProviderOperation) -> ProviderResult[ProviderValue]:  # noqa: PLR0912,PLR0915 — closed provider union dispatch
        if isinstance(operation, CreateCustomerOperation):
            existing = self._customers_by_correlation.get(operation.correlation_id)
            if existing is not None:
                return self._success(operation, value=existing)
            customer_id = self._customers_by_email.setdefault(
                operation.email.lower(), f"cust_memory_{len(self._customers_by_email) + 1}"
            )
            customer = ProviderCustomer(provider_customer_id=customer_id, email=operation.email)
            self._customers_by_correlation[operation.correlation_id] = customer
            return self._success(
                operation,
                value=customer,
            )

        if isinstance(operation, FindCustomerByEmailOperation):
            customer_id = self._customers_by_email.get(operation.email.lower())
            if customer_id is None:
                return Failure(
                    ProviderNotFoundError(
                        operation=operation.kind, correlation_id=operation.correlation_id
                    )
                )
            return self._success(
                operation,
                value=ProviderCustomer(provider_customer_id=customer_id, email=operation.email),
            )

        if isinstance(operation, CreateSubscriptionOperation):
            existing = self._subscriptions_by_correlation.get(operation.correlation_id)
            if existing is not None:
                return self._success(operation, value=existing)
            number = len(self._subscriptions_by_id) + 1
            state = ProviderSubscriptionState(
                provider_subscription_id=f"sub_memory_{number}",
                status="created",
                payment_url=f"https://example.test/subscriptions/{number}",
                plan_id=operation.plan_id,
            )
            self._subscriptions_by_correlation[operation.correlation_id] = state
            self._subscriptions_by_id[state.provider_subscription_id] = state
            return self._success(operation, value=state)

        if isinstance(operation, FindSubscriptionByCorrelationOperation):
            state = self._subscriptions_by_correlation.get(operation.correlation_id)
            if state is None:
                return self._manual_review(operation)
            return self._success(operation, value=state)

        if isinstance(operation, ObserveSubscriptionOperation):
            state = self._subscriptions_by_id.get(operation.provider_subscription_id)
            if state is None:
                return Failure(
                    ProviderNotFoundError(
                        operation=operation.kind, correlation_id=operation.correlation_id
                    )
                )
            return self._success(operation, value=state)

        if isinstance(operation, ListPaymentsOperation):
            return self._success(operation, value=ProviderPaymentBatch(payments=self._payments))

        if isinstance(operation, CancelSubscriptionOperation):
            return self._update_subscription(operation, status="cancelled")

        if isinstance(operation, PauseSubscriptionOperation):
            return self._update_subscription(operation, status="paused")

        if isinstance(operation, ResumeSubscriptionOperation):
            return self._update_subscription(operation, status="active")

        if isinstance(operation, ChangeSubscriptionPlanOperation):
            return self._update_subscription(operation, status="active")

        if isinstance(operation, CreatePaymentLinkOperation):
            existing = self._payment_links_by_correlation.get(operation.correlation_id)
            if existing is not None:
                return self._success(operation, value=existing)
            number = len(self._payment_links_by_correlation) + 1
            link = ProviderPaymentLink(
                provider_payment_link_id=f"plink_memory_{number}",
                short_url=f"https://example.test/pay/{number}",
                status="created",
            )
            self._payment_links_by_correlation[operation.correlation_id] = link
            return self._success(operation, value=link)

        if isinstance(operation, FindPaymentLinkByCorrelationOperation):
            link = self._payment_links_by_correlation.get(operation.correlation_id)
            if link is None:
                return self._manual_review(operation)
            return self._success(operation, value=link)

        if isinstance(operation, CreatePlanOperation):
            existing = self._plans_by_correlation.get(operation.correlation_id)
            if existing is not None:
                return self._success(operation, value=existing)
            plan = ProviderPlan(
                provider_plan_id=f"plan_memory_{len(self._plans_by_correlation) + 1}",
                name=operation.name,
            )
            self._plans_by_correlation[operation.correlation_id] = plan
            return self._success(operation, value=plan)

        if isinstance(operation, FindPlanByCorrelationOperation):
            plan = self._plans_by_correlation.get(operation.correlation_id)
            if plan is None:
                return self._manual_review(operation)
            return self._success(operation, value=plan)

        if isinstance(operation, CreateRefundOperation):
            existing = self._refunds_by_correlation.get(operation.correlation_id)
            if existing is not None:
                return self._success(operation, value=existing)
            number = len(self._refunds_by_correlation) + 1
            receipt = ProviderReceipt(
                provider_refund_id=f"rfnd_memory_{number}",
                payment_id=operation.payment_id,
                amount=operation.amount,
            )
            self._refunds_by_correlation[operation.correlation_id] = receipt
            return self._success(operation, value=receipt)

        if isinstance(operation, FindRefundByCorrelationOperation):
            receipt = self._refunds_by_correlation.get(operation.correlation_id)
            if receipt is None:
                return self._manual_review(operation)
            return self._success(operation, value=receipt)

        assert_never(operation)

    def _update_subscription(
        self,
        operation: CancelSubscriptionOperation
        | PauseSubscriptionOperation
        | ResumeSubscriptionOperation
        | ChangeSubscriptionPlanOperation,
        *,
        status: str,
    ) -> ProviderResult[ProviderValue]:
        state = self._subscriptions_by_id.get(operation.provider_subscription_id)
        if state is None:
            return Failure(
                ProviderNotFoundError(
                    operation=operation.kind, correlation_id=operation.correlation_id
                )
            )
        updated = ProviderSubscriptionState(
            provider_subscription_id=state.provider_subscription_id,
            status=status,
            current_period_start=state.current_period_start,
            current_period_end=state.current_period_end,
            ended_at=state.ended_at,
            payment_url=state.payment_url,
            cancel_at_cycle_end=(
                operation.cancel_at_cycle_end
                if isinstance(operation, CancelSubscriptionOperation)
                else state.cancel_at_cycle_end
            ),
            plan_id=(
                operation.plan_id
                if isinstance(operation, ChangeSubscriptionPlanOperation)
                else state.plan_id
            ),
        )
        self._subscriptions_by_id[updated.provider_subscription_id] = updated
        for correlation, candidate in self._subscriptions_by_correlation.items():
            if candidate.provider_subscription_id == updated.provider_subscription_id:
                self._subscriptions_by_correlation[correlation] = updated
        return self._success(operation, value=updated)

    @staticmethod
    def _success(
        operation: ProviderOperation, *, value: ProviderValue
    ) -> ProviderResult[ProviderValue]:
        return Success(
            ProviderSuccess(
                operation=operation.kind,
                correlation_id=operation.correlation_id,
                value=value,
            )
        )

    @staticmethod
    def _manual_review(operation: ProviderOperation) -> ProviderResult[ProviderValue]:
        return Failure(
            ProviderManualReviewRequiredError(
                operation=operation.kind,
                correlation_id=operation.correlation_id,
            )
        )
