"""HTTP contracts for the subscription lifecycle mutation surface.

These tests deliberately stop below the network/database boundary: the route,
request middleware, dependency override seam, service-shaped object, durable
coordinator, and provider port are real.  The provider and lifecycle
persistence are in-memory doubles, so a focused run never needs a live Razorpay
account or an online Alembic migration.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from returns.result import Failure, Success

from app.features.auth.api.http.dependencies import get_current_verified_user
from app.features.billing.providers import (
    CreateRefundOperation,
    CreateSubscriptionOperation,
    InMemoryBillingProvider,
    ProviderOperationKind,
    ProviderOutcomeUnknownError,
)
from app.features.billing.subscriptions.coordinator import (
    InMemoryLifecyclePersistence,
    LifecycleRecord,
    SubscriptionLifecycleCoordinator,
)
from app.features.billing.subscriptions.dependencies import get_subscription_service
from app.features.billing.subscriptions.dto import (
    SubscriptionResponse,
)
from app.features.billing.subscriptions.lifecycle import (
    ActiveState,
    CancelSubscription,
    ChangeSubscriptionPlan,
    CommandMetadata,
    CreatedState,
    CreateSubscription,
    PauseSubscription,
    PolicyContext,
    RemoteSyncState,
    ResumeSubscription,
    SubscriptionSnapshot,
)
from app.features.billing.subscriptions.models import SubscriptionStatus
from app.features.billing.subscriptions.repository import SqlAlchemyLifecyclePersistence
from app.features.billing.subscriptions.router import router as subscriptions_router
from app.features.billing.subscriptions.service import SubscriptionService
from app.middleware import RequestStateLoggingMiddleware

if TYPE_CHECKING:
    from typing import Any
    from uuid import UUID

    from app.features.billing.subscriptions.dto import (
        PlanChangeDTO,
        SubscriptionCancelDTO,
        SubscriptionCreateDTO,
        SubscriptionPauseDTO,
    )


pytestmark = pytest.mark.integration

USER_ID = "user-http-contract"
PLAN_A = uuid4()
PLAN_B = uuid4()
PERIOD_START = datetime(2026, 10, 1, tzinfo=UTC)
PERIOD_END = datetime(2026, 11, 1, tzinfo=UTC)
CREATED_AT = datetime(2026, 10, 1, tzinfo=UTC)


@dataclass
class _Plan:
    id: UUID
    name: str
    amount: int
    razorpay_plan_id: str
    currency: str = "INR"
    interval: str = "monthly"

    is_active: bool = True
    trial_period_days: int = 0


@dataclass
class _HttpState:
    id: UUID
    snapshot: SubscriptionSnapshot
    provider_subscription_id: str | None
    provider_customer_id: str | None
    created_at: datetime = CREATED_AT


class _InMemoryCreditService:
    """No-op credit collaborator for the production composition root.

    The HTTP harness patches the mutation methods after construction, but the
    real dependency factory still requires this collaborator.  Returning a
    successful result keeps any downgrade-credit path safe if the service is
    exercised directly in a future contract test.
    """

    async def grant_credit_on_downgrade(self, *args: object, **kwargs: object) -> Success[None]:
        del args, kwargs
        return Success(None)


class _UnknownAfterPaymentLinkProvider(InMemoryBillingProvider):
    """Persist a payment link, then report an ambiguous provider outcome once."""

    def __init__(self) -> None:
        super().__init__()
        self._unknown_payment_link = True

    async def execute(self, operation: Any) -> Any:
        if (
            self._unknown_payment_link
            and operation.kind is ProviderOperationKind.CREATE_PAYMENT_LINK
        ):
            self._unknown_payment_link = False
            result = await super().execute(operation)
            assert isinstance(result, Success)
            return Failure(ProviderOutcomeUnknownError(operation.kind, operation.correlation_id))
        return await super().execute(operation)


class _MutationService:
    """Small service-shaped HTTP adapter backed by the real coordinator port."""

    def __init__(self, *, active: bool = False, unknown_payment_link: bool = False) -> None:
        self.provider = (
            _UnknownAfterPaymentLinkProvider()
            if unknown_payment_link
            else InMemoryBillingProvider()
        )
        self.plans = {
            str(PLAN_A): _Plan(PLAN_A, "Starter", 1_000, "rp_plan_a"),
            str(PLAN_B): _Plan(PLAN_B, "Growth", 2_000, "rp_plan_b"),
        }
        self.persistence = InMemoryLifecyclePersistence()
        self.coordinator = SubscriptionLifecycleCoordinator(self.persistence, self.provider)
        self.calls: list[tuple[str, str]] = []
        self.refund_intents: list[str] = []
        self.wired_service: object | None = None
        self._state = self._seed(active=active)

    def _seed(self, *, active: bool) -> _HttpState:
        subscription_id = uuid4()
        provider_subscription_id: str | None = None
        provider_customer_id: str | None = None
        if active:
            provider_customer_id = "cust_http_1"
            provider_subscription_id = asyncio.run(
                self._seed_provider_subscription(provider_customer_id)
            )
            snapshot = SubscriptionSnapshot(
                id=str(subscription_id),
                user_id=USER_ID,
                plan_id=str(PLAN_A),
                state=ActiveState(PERIOD_START, PERIOD_END),
                version=4,
                remote_sync=RemoteSyncState.SYNCED,
                observed_plan_id=str(PLAN_A),
            )
        else:
            snapshot = SubscriptionSnapshot(
                id=str(subscription_id),
                user_id=USER_ID,
                plan_id=str(PLAN_A),
                state=CreatedState(),
                version=0,
                remote_sync=RemoteSyncState.PENDING,
            )
        self.persistence.seed(
            LifecycleRecord(
                snapshot=snapshot,
                customer_email="http@example.test",
                provider_customer_id=provider_customer_id,
                provider_subscription_id=provider_subscription_id,
                provider_plan_id="rp_plan_a",
                desired_provider_plan_id="rp_plan_a",
            )
        )
        return _HttpState(subscription_id, snapshot, provider_subscription_id, provider_customer_id)

    async def _seed_provider_subscription(self, customer_id: str) -> str:
        result = await self.provider.execute(
            CreateSubscriptionOperation(plan_id="rp_plan_a", customer_id=customer_id, total_count=0, quantity=1, customer_notify=True, correlation_id="seed-http")
        )
        assert isinstance(result, Success)
        value = result.unwrap().value
        return value.provider_subscription_id  # type: ignore[union-attr]

    def _record(self) -> _HttpState:
        record = self.persistence.records[str(self._state.id)]
        return replace(
            self._state,
            snapshot=record.snapshot,
            provider_subscription_id=record.provider_subscription_id,
            provider_customer_id=record.provider_customer_id,
        )

    def _response(
        self, *, plan: _Plan | None = None, payment_url: str | None = None
    ) -> SubscriptionResponse:
        state = self._record()
        snapshot = state.snapshot
        lifecycle_state = snapshot.state
        status = lifecycle_state.kind.value
        cancel_intent = snapshot.cancellation_intent
        period_end = getattr(lifecycle_state, "period_end", None)
        pause_start = getattr(lifecycle_state, "start", None)
        pause_end = getattr(lifecycle_state, "end", None)
        entitlement_until = period_end if snapshot.grants_paid_entitlement else None
        if plan is None:
            plan = self.plans.get(snapshot.plan_id)
        return SubscriptionResponse(
            id=str(state.id),
            user_id=USER_ID,
            plan_id=snapshot.plan_id,
            plan=(
                {
                    "id": str(plan.id),
                    "name": plan.name,
                    "amount": plan.amount,
                    "currency": plan.currency,
                    "interval": plan.interval,
                }
                if plan is not None
                else None
            ),
            razorpay_subscription_id=state.provider_subscription_id,
            status=status,
            current_period_start=getattr(lifecycle_state, "period_start", None),
            current_period_end=period_end,
            cancel_at_period_end=bool(cancel_intent and cancel_intent.at_period_end),
            pause_start=pause_start,
            pause_end=pause_end,
            version=snapshot.version,
            remote_sync=snapshot.remote_sync.value,
            observed_plan_id=snapshot.observed_plan_id,
            entitlement_until=entitlement_until,
            cancellation_requested_at=(cancel_intent.requested_at if cancel_intent else None),
            payment_url=payment_url,
            created_at=state.created_at,
            updated_at=state.created_at,
        )

    async def create_subscription(
        self, user_id: str, dto: SubscriptionCreateDTO, *, idempotency_key: str | None = None
    ) -> Any:
        key = idempotency_key or "missing-key"
        self.calls.append(("create", key))
        result = await self.coordinator.apply(
            CreateSubscription(
                CommandMetadata(request_id=key, expected_version=self._record().snapshot.version), dto.plan_id
            ),
            key,
            subscription_id=str(self._state.id),
            context=PolicyContext(now=datetime.now(tz=UTC)),
        )
        if isinstance(result, Failure):
            return result
        return Success(self._response(payment_url="https://example.test/subscriptions/1"))

    async def cancel_subscription(
        self,
        user_id: str,
        subscription_id: str,
        dto: SubscriptionCancelDTO,
        *,
        idempotency_key: str | None = None,
    ) -> Any:
        key = idempotency_key or "missing-key"
        self.calls.append(("cancel", key))
        result = await self.coordinator.apply(
            CancelSubscription(
                CommandMetadata(request_id=key, expected_version=self._record().snapshot.version),
                cancel_at_period_end=dto.cancel_at_period_end,
            ),
            key,
            subscription_id=subscription_id,
            context=PolicyContext(now=datetime.now(tz=UTC)),
        )
        if isinstance(result, Failure):
            return result
        if not dto.cancel_at_period_end:
            refund_key = f"{key}:refund:payment-http"
            refund = await self.coordinator.execute_provider_operation(
                CreateRefundOperation("pay_http", 500, f"{subscription_id}:refund"),
                subscription_id=subscription_id,
                idempotency_key=refund_key,
            )
            if isinstance(refund, Success):
                self.refund_intents.append(refund_key)
        return Success(self._response())

    async def pause_subscription(
        self,
        user_id: str,
        subscription_id: str,
        dto: SubscriptionPauseDTO,
        *,
        idempotency_key: str | None = None,
    ) -> Any:
        key = idempotency_key or "missing-key"
        self.calls.append(("pause", key))
        result = await self.coordinator.apply(
            PauseSubscription(
                CommandMetadata(request_id=key, expected_version=self._record().snapshot.version),
                duration_days=dto.pause_duration_days,
            ),
            key,
            subscription_id=subscription_id,
            context=PolicyContext(now=datetime.now(tz=UTC)),
        )
        return result.map(lambda _: self._response())

    async def resume_subscription(
        self, user_id: str, subscription_id: str, *, idempotency_key: str | None = None
    ) -> Any:
        key = idempotency_key or "missing-key"
        self.calls.append(("resume", key))
        result = await self.coordinator.apply(
            ResumeSubscription(
                CommandMetadata(request_id=key, expected_version=self._record().snapshot.version)
            ),
            key,
            subscription_id=subscription_id,
            context=PolicyContext(now=datetime.now(tz=UTC)),
        )
        return result.map(lambda _: self._response())

    async def change_plan(
        self,
        user_id: str,
        subscription_id: str,
        dto: PlanChangeDTO,
        *,
        idempotency_key: str | None = None,
    ) -> Any:
        key = idempotency_key or "missing-key"
        self.calls.append(("change-plan", key))
        record = self.persistence.records[subscription_id]
        self.persistence.records[subscription_id] = record.model_copy(
            update={"desired_provider_plan_id": "rp_plan_b"}
        )
        result = await self.coordinator.apply(
            ChangeSubscriptionPlan(
                metadata=CommandMetadata(
                    request_id=key, expected_version=self._record().snapshot.version
                ),
                new_plan_id=dto.new_plan_id,
                effective_at=dto.effective_date or datetime.now(tz=UTC),
            ),
            key,
            subscription_id=subscription_id,
            context=PolicyContext(
                now=datetime.now(tz=UTC),
                current_plan_amount=1_000,
                target_plan_amount=2_000,
                current_interval="monthly",
                target_interval="monthly",
                target_currency=self.plans[dto.new_plan_id].currency,
                target_provider_plan_id=self.plans[dto.new_plan_id].razorpay_plan_id,
            ),
        )
        if isinstance(result, Failure):
            return result
        return Success(self._response(plan=self.plans[dto.new_plan_id]))


def _app(service: _MutationService) -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestStateLoggingMiddleware)
    app.include_router(subscriptions_router, prefix="/api/v2")
    user = SimpleNamespace(id=USER_ID, is_active=True, is_verified=True)
    app.dependency_overrides[get_current_verified_user] = lambda: user

    async def fake_session() -> object:
        return object()

    async def fake_payment_service() -> object:
        return object()

    async def fake_provider() -> InMemoryBillingProvider:
        return service.provider

    async def fake_credit_service() -> _InMemoryCreditService:
        return _InMemoryCreditService()

    async def wired_service(
        session: object = Depends(fake_session),
        payment_service: object = Depends(fake_payment_service),
        provider: InMemoryBillingProvider = Depends(fake_provider),
        credit_service: _InMemoryCreditService = Depends(fake_credit_service),
    ) -> object:
        # Run the production composition root, including its SQL persistence
        # and coordinator construction.  Only the service methods are patched
        # to the in-memory HTTP harness; no route dependency is replaced with a
        # raw provider/client escape hatch.
        production_service = await get_subscription_service(
            session, payment_service, provider, credit_service
        )
        for name in (
            "create_subscription",
            "cancel_subscription",
            "pause_subscription",
            "resume_subscription",
            "change_plan",
        ):
            setattr(production_service, name, getattr(service, name))
        service.wired_service = production_service
        return production_service

    app.dependency_overrides[get_subscription_service] = wired_service
    return app


def _data(response: Any) -> dict[str, Any]:
    payload = response.json()
    assert payload["success"] is True
    assert payload["statusCode"] == response.status_code
    assert payload["error"] is None
    return payload["data"]


def test_create_route_preserves_status_shape_and_keyless_correlation_idempotency() -> None:
    service = _MutationService()
    with TestClient(_app(service)) as client:
        headers = {"X-Correlation-ID": "http-create-correlation"}
        body = {"plan_id": str(PLAN_A), "customer_email": "http@example.test"}
        first = client.post("/api/v2/subscriptions", json=body, headers=headers)
        second = client.post("/api/v2/subscriptions", json=body, headers=headers)

    assert first.status_code == 201
    assert second.status_code == 201
    assert _data(first)["status"] == SubscriptionStatus.CREATED.value
    assert _data(first)["entitlementUntil"] is None
    assert first.json()["request"]["correlationId"] == "http-create-correlation"
    assert second.json()["request"]["correlationId"] == "http-create-correlation"
    assert service.calls == [("create", "http-create-correlation")] * 2
    assert isinstance(service.wired_service, SubscriptionService)
    assert isinstance(
        service.wired_service.lifecycle_coordinator._persistence,
        SqlAlchemyLifecyclePersistence,
    )
    assert service.wired_service.lifecycle_coordinator._provider is service.provider
    assert [call.operation.kind for call in service.provider.calls].count(
        ProviderOperationKind.CREATE_SUBSCRIPTION
    ) == 1


def test_cancel_at_period_end_retains_entitlement_and_durable_intent() -> None:
    service = _MutationService(active=True)
    with TestClient(_app(service)) as client:
        response = client.post(
            f"/api/v2/subscriptions/{service._state.id}/cancel",
            json={"cancel_at_period_end": True, "reason": "too expensive"},
            headers={"Idempotency-Key": "cancel-period-end"},
        )

    data = _data(response)
    assert response.status_code == 200
    assert data["status"] == SubscriptionStatus.ACTIVE.value
    assert data["cancelAtPeriodEnd"] is True
    assert data["entitlementUntil"] == PERIOD_END.isoformat().replace("+00:00", "Z")
    assert data["cancellationRequestedAt"] is not None
    assert data["remoteSync"] == RemoteSyncState.PENDING.value


def test_immediate_cancel_records_refund_intent_and_uses_existing_response_contract() -> None:
    service = _MutationService(active=True)
    with TestClient(_app(service)) as client:
        response = client.post(
            f"/api/v2/subscriptions/{service._state.id}/cancel",
            json={"cancel_at_period_end": False},
            headers={"Idempotency-Key": "cancel-now"},
        )

    assert response.status_code == 200
    data = _data(response)
    assert data["status"] == SubscriptionStatus.CANCELLED.value
    assert data["entitlementUntil"] is None
    assert service.refund_intents == ["cancel-now:refund:payment-http"]
    assert any(
        call.operation.kind is ProviderOperationKind.CREATE_REFUND
        for call in service.provider.calls
    )


def test_pause_and_resume_expose_desired_state_and_pending_remote_sync() -> None:
    service = _MutationService(active=True)
    with TestClient(_app(service)) as client:
        paused = client.post(
            f"/api/v2/subscriptions/{service._state.id}/pause",
            json={"pause_duration_days": 7},
            headers={"Idempotency-Key": "pause-1"},
        )
        resumed = client.post(
            f"/api/v2/subscriptions/{service._state.id}/resume",
            headers={"Idempotency-Key": "resume-1"},
        )

    paused_data = _data(paused)
    assert paused.status_code == 200
    assert paused_data["status"] == SubscriptionStatus.PAUSED.value
    assert paused_data["pauseStart"] is not None
    assert paused_data["pauseEnd"] is not None
    assert paused_data["remoteSync"] == RemoteSyncState.PENDING.value
    assert resumed.status_code == 200
    assert _data(resumed)["status"] == SubscriptionStatus.ACTIVE.value


def test_change_plan_keeps_desired_and_observed_plans_distinct() -> None:
    service = _MutationService(active=True)
    with TestClient(_app(service)) as client:
        response = client.post(
            f"/api/v2/subscriptions/{service._state.id}/change-plan",
            json={"new_plan_id": str(PLAN_B)},
            headers={"Idempotency-Key": "change-plan-1"},
        )

    data = _data(response)
    assert response.status_code == 200
    assert data["planId"] == str(PLAN_B)
    assert data["observedPlanId"] == str(PLAN_A)
    assert data["remoteSync"] == RemoteSyncState.PENDING.value


def test_unknown_provider_outcome_reconciles_without_reissuing_mutation() -> None:
    service = _MutationService(active=True, unknown_payment_link=True)
    with TestClient(_app(service)) as client:
        path = f"/api/v2/subscriptions/{service._state.id}/change-plan"
        first = client.post(
            path,
            json={"new_plan_id": str(PLAN_B)},
            headers={"Idempotency-Key": "change-unknown"},
        )
        calls_after_first = len(service.provider.calls)
        second = client.post(
            path,
            json={"new_plan_id": str(PLAN_B)},
            headers={"Idempotency-Key": "change-unknown"},
        )

    assert first.status_code == second.status_code == 200
    assert len(service.provider.calls) == calls_after_first
    operation_kinds = [call.operation.kind for call in service.provider.calls]
    assert operation_kinds.count(ProviderOperationKind.CREATE_PAYMENT_LINK) == 1
    assert operation_kinds.count(ProviderOperationKind.FIND_PAYMENT_LINK_BY_CORRELATION) == 1
    assert ProviderOperationKind.CHANGE_SUBSCRIPTION_PLAN not in operation_kinds
    first_data = _data(first)
    second_data = _data(second)
    assert first_data["planId"] == second_data["planId"] == str(PLAN_B)
    assert first_data["status"] == second_data["status"] == SubscriptionStatus.ACTIVE.value
    assert (
        first_data["observedPlanId"]
        == second_data["observedPlanId"]
        == service.plans[str(PLAN_A)].razorpay_plan_id
    )
    pending_upgrade = service._record().snapshot.pending_upgrade
    assert pending_upgrade is not None
    assert pending_upgrade.target_plan_id == str(PLAN_B)
    assert pending_upgrade.currency == service.plans[str(PLAN_B)].currency


def test_production_dependency_factory_constructs_sql_persistence_and_coordinator() -> None:
    service = asyncio.run(
        get_subscription_service(
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            InMemoryBillingProvider(),
            _InMemoryCreditService(),  # type: ignore[arg-type]
        )
    )

    assert isinstance(service.lifecycle_coordinator, SubscriptionLifecycleCoordinator)
    assert (
        type(service.lifecycle_coordinator._persistence).__name__
        == "SqlAlchemyLifecyclePersistence"
    )
    assert service.razorpay is None
