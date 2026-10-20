"""Worker lifecycle jobs cross the coordinator seam for status transitions."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from returns.result import Success

import app.features.billing.jobs as billing_jobs
from app.features.billing.dunning.service import DunningService
from app.features.billing.providers import (
    ObserveSubscriptionOperation,
    ProviderOperationKind,
    ProviderSubscriptionState,
    ProviderSuccess,
)
from app.features.billing.subscriptions.coordinator import CoordinatorStatus
from app.features.billing.subscriptions.lifecycle import SubscriptionCharged, SubscriptionHalted


def test_worker_lifecycle_paths_do_not_bypass_coordinator_with_status_writes() -> None:
    jobs_source = Path("src/app/features/billing/jobs.py").read_text(encoding="utf-8")
    dunning_source = Path("src/app/features/billing/dunning/service.py").read_text(encoding="utf-8")

    assert "RazorpayClient" not in jobs_source
    assert "RazorpayClient" not in dunning_source
    assert ".update_status(" not in jobs_source
    assert ".update_status(" not in dunning_source
    assert "SubscriptionLifecycleCoordinator" in jobs_source
    assert "SubscriptionLifecycleCoordinator" in dunning_source
    assert ".observe(" in jobs_source
    assert ".apply(" in jobs_source


@pytest.mark.asyncio
async def test_terminal_dunning_observes_provider_before_coordinator_halt() -> None:
    subscription = SimpleNamespace(
        id="sub-1",
        status="past_due",
        metadata_={},
        retry_count=2,
        max_retries=3,
        version=4,
        razorpay_subscription_id="rzp_sub_1",
    )
    updated = SimpleNamespace(id="sub-1", status="halted", retry_count=3)
    subscriptions = SimpleNamespace(
        update_with_lock=AsyncMock(return_value=Success(updated)),
    )
    provider = SimpleNamespace(
        execute=AsyncMock(
            return_value=Success(
                ProviderSuccess(
                    operation=ProviderOperationKind.OBSERVE_SUBSCRIPTION,
                    correlation_id="observe",
                    value=ProviderSubscriptionState(
                        provider_subscription_id="rzp_sub_1", status="active"
                    ),
                )
            )
        )
    )
    coordinator = SimpleNamespace(
        observe=AsyncMock(
            return_value=Success(SimpleNamespace(snapshot=SimpleNamespace(version=5)))
        )
    )
    service = DunningService(
        session=AsyncMock(),
        subscriptions=subscriptions,
        plans=AsyncMock(),
        audit=AsyncMock(),
        provider=provider,
        lifecycle_coordinator=coordinator,
    )
    service._attempt_charge = AsyncMock(return_value=Success({"status": "skipped"}))
    service._next_retry_at = lambda _subscription, *, now: now

    result = await service.execute_retry(subscription)

    assert isinstance(result, Success)
    observed_operation = provider.execute.await_args.args[0]
    assert isinstance(observed_operation, ObserveSubscriptionOperation)
    halt_event = coordinator.observe.await_args.args[0]
    assert isinstance(halt_event, SubscriptionHalted)
    subscriptions.update_with_lock.assert_awaited_once()


@pytest.mark.asyncio
async def test_renewal_provider_observation_is_forwarded_to_coordinator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 10, 20, 12, tzinfo=UTC)
    subscription = SimpleNamespace(
        id="sub-renewal",
        version=7,
        razorpay_subscription_id="rzp_sub_renewal",
        current_period_end=now.replace(day=19),
    )
    session = SimpleNamespace(
        execute=AsyncMock(
            return_value=SimpleNamespace(
                scalars=lambda: SimpleNamespace(all=lambda: [subscription])
            )
        )
    )
    audit = SimpleNamespace(create=AsyncMock())
    provider = SimpleNamespace(
        execute=AsyncMock(
            return_value=Success(
                ProviderSuccess(
                    operation=ProviderOperationKind.OBSERVE_SUBSCRIPTION,
                    correlation_id="observation",
                    value=ProviderSubscriptionState(
                        provider_subscription_id="rzp_sub_renewal",
                        status="active",
                        current_period_start=now.replace(day=1),
                        current_period_end=now.replace(day=30),
                    ),
                )
            )
        )
    )
    observed_inputs: list[object] = []

    class FakeCoordinator:
        def __init__(self, _persistence: object, _provider: object) -> None:
            pass

        async def observe(self, input_: object, **_kwargs: object) -> object:
            observed_inputs.append(input_)
            return Success(SimpleNamespace(status=CoordinatorStatus.RECONCILED))

    @asynccontextmanager
    async def item_session(_factory: object):
        yield SimpleNamespace()

    monkeypatch.setattr(billing_jobs, "_current_utc", lambda: now)
    monkeypatch.setattr(billing_jobs, "_audit_repo", lambda _session: audit)
    monkeypatch.setattr(billing_jobs, "independent_session", item_session)
    monkeypatch.setattr(billing_jobs, "SqlAlchemyLifecyclePersistence", lambda _session: object())
    monkeypatch.setattr(billing_jobs, "SubscriptionLifecycleCoordinator", FakeCoordinator)

    result = await billing_jobs.renewal_job(session, object(), provider=provider)

    assert result == {"checked": 1, "renewed": 1}
    operation = provider.execute.await_args.args[0]
    assert isinstance(operation, ObserveSubscriptionOperation)
    assert isinstance(observed_inputs[0], SubscriptionCharged)
