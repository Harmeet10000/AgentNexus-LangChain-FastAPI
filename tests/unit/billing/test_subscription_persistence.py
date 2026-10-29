"""Public-interface coverage for subscription convergence persistence."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from returns.result import Success

from app.features.billing.subscriptions.lifecycle import (
    ActiveState,
    CancellationIntent,
    RemoteSyncState,
    SubscriptionSnapshot,
)
from app.features.billing.subscriptions.models import (
    ProviderOperationRecord,
    Subscription,
)
from app.features.billing.subscriptions.repository import (
    ProviderOperationInput,
    ProviderOperationUpdate,
    SubscriptionLifecycleRepository,
)


def _row(*, version: int = 3) -> Subscription:
    return Subscription(
        id=uuid4(),
        user_id="user-1",
        plan_id=uuid4(),
        status="active",
        version=version,
        current_period_start=datetime(2026, 1, 1, tzinfo=UTC),
        current_period_end=datetime(2026, 2, 1, tzinfo=UTC),
        observed_plan_id=uuid4(),
        remote_sync="pending",
        cancellation_requested_at=datetime(2026, 1, 15, tzinfo=UTC),
        cancel_at_period_end=True,
    )


def _result(value: object) -> SimpleNamespace:
    return SimpleNamespace(scalar_one_or_none=lambda: value)


@pytest.mark.asyncio
async def test_public_snapshot_interface_round_trips_desired_observed_and_cancel_intent() -> None:
    row = _row()
    session = SimpleNamespace(execute=AsyncMock(return_value=_result(row)))
    repository = SubscriptionLifecycleRepository(session)

    loaded = await repository.load_snapshot(row.id)

    assert isinstance(loaded, Success)
    snapshot = loaded.unwrap()
    assert snapshot.plan_id == str(row.plan_id)
    assert snapshot.observed_plan_id == str(row.observed_plan_id)
    assert snapshot.remote_sync is RemoteSyncState.PENDING
    assert snapshot.cancellation_intent == CancellationIntent(
        requested_at=row.cancellation_requested_at,
        at_period_end=True,
    )


@pytest.mark.asyncio
async def test_public_operation_interface_deduplicates_and_sanitizes_update_payload() -> None:
    record = ProviderOperationRecord(
        id=uuid4(),
        correlation_id="corr-1",
        idempotency_key="idem-1",
        operation_kind="create_subscription",
    )
    session = SimpleNamespace(
        add=lambda _: None,
        flush=AsyncMock(),
        execute=AsyncMock(return_value=_result(record)),
    )
    repository = SubscriptionLifecycleRepository(session)

    created = await repository.create_provider_operation(
        ProviderOperationInput(
            correlation_id="corr-1",
            idempotency_key="idem-1",
            operation_kind="create_subscription",
            request_data={"plan_id": "plan-1"},
        )
    )
    assert isinstance(created, Success)
    assert created.unwrap().request_data == {"plan_id": "plan-1"}
    found = await repository.find_provider_operation_by_correlation("corr-1")
    assert isinstance(found, Success)
    assert found.unwrap() is record

    updated = await repository.update_provider_operation(
        record.id,
        ProviderOperationUpdate(
            state="manual-review",
            attempts=2,
            provider_resource_id="sub_remote_1",
            error_message="  token\nleak  ",
            observation_data={"status": "created"},
        ),
    )
    assert isinstance(updated, Success)
    statement = str(session.execute.await_args.args[0])
    assert "provider_operation_records" in statement
    values = session.execute.await_args.args[0]._values
    by_name = {column.key: value for column, value in values.items()}
    assert by_name["state"].value == "manual_review"
    assert by_name["error_message"].value == "token leak"


def test_model_aliases_keep_legacy_reads_and_make_convergence_facts_queryable() -> None:
    assert Subscription.desired_plan_id.property.key == "plan_id"
    assert Subscription.lifecycle_version.property.key == "version"
    assert Subscription.provider_observed_plan_id.property.key == "observed_plan_id"
    assert Subscription.remote_sync_state.property.key == "remote_sync"
    assert ProviderOperationRecord.provider_operation_state.property.key == "state"
    assert ProviderOperationRecord.normalized_request.property.key == "request_data"


def test_model_and_migration_keep_pending_defaults_in_lockstep() -> None:
    remote_sync = Subscription.__table__.c.remote_sync
    operation_state = ProviderOperationRecord.__table__.c.state
    assert remote_sync.default is not None
    assert remote_sync.default.arg == "pending"
    assert remote_sync.server_default is not None
    assert str(remote_sync.server_default.arg) == "pending"
    assert operation_state.default is not None
    assert operation_state.default.arg == "pending"
    migration = (
        Path(__file__).parents[3]
        / "src/alembic/versions/0023_subscription_convergence_persistence.py"
    )
    text = migration.read_text(encoding="utf-8")
    assert "server_default=sa.text(\"'pending'\")" in text
    assert 'server_default="pending"' in text


def test_snapshot_save_preserves_period_end_entitlement_boundary() -> None:
    snapshot = SubscriptionSnapshot(
        id=str(uuid4()),
        user_id="user-1",
        plan_id=str(uuid4()),
        state=ActiveState(
            period_start=datetime(2026, 1, 1, tzinfo=UTC),
            period_end=datetime(2026, 2, 1, tzinfo=UTC),
        ),
        version=4,
        remote_sync=RemoteSyncState.PENDING,
        cancellation_intent=CancellationIntent(
            requested_at=datetime(2026, 1, 15, tzinfo=UTC), at_period_end=True
        ),
    )
    assert snapshot.grants_paid_entitlement is True
