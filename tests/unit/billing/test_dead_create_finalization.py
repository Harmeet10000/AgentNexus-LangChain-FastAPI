"""G10: a dead-create subscription must have an exit to terminality.

A row whose provider create returned permanent_failure sits in ``created``
forever — the cancel endpoint refused ``created -> cancelled``, recovery
ignores provider-less rows, and the duplicate guard blocks any new
subscription for the user+plan. The service now local-finalizes exactly
that shape (permanent_failure, no provider binding): no provider call,
no refund — nothing moved. A merely-pending create must reconcile, never
terminalize blind.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from returns.result import Failure, Success

from app.features.billing.subscriptions import service as subscription_service_module
from app.features.billing.subscriptions.service import SubscriptionService
from app.features.billing.subscriptions.dto import SubscriptionCancelDTO
from app.features.billing.subscriptions.exceptions import (
    SubscriptionInvalidTransitionError,
)
from app.features.billing.subscriptions.lifecycle import RemoteSyncState
from app.features.billing.subscriptions.models import Subscription

PLAN_ID = "00000000-0000-0000-0000-00000000000c"


def _row(*, status: str, remote_sync: str, provider_id: str | None) -> Subscription:
    return Subscription(
        id="00000000-0000-0000-0000-00000000000a",
        user_id="00000000-0000-0000-0000-00000000000b",
        plan_id=PLAN_ID,
        status=status,
        remote_sync=remote_sync,
        razorpay_subscription_id=provider_id,
        version=3,
        created_at=datetime.now(tz=UTC),
        updated_at=datetime.now(tz=UTC),
    )


def _make_service(
    monkeypatch: pytest.MonkeyPatch, *, row: Subscription, finalized: Subscription
) -> tuple[SubscriptionService, Any]:
    """A service with the repository seam and the response mapper stubbed."""
    from unittest.mock import AsyncMock

    repo = AsyncMock()
    repo.update_status = AsyncMock(return_value=Success(finalized))
    service = SubscriptionService.__new__(SubscriptionService)
    service.subscriptions = repo
    service._get_owned_subscription = AsyncMock(return_value=Success(row))
    service._load_plan = AsyncMock(
        return_value=Success(
            SimpleNamespace(
                id=PLAN_ID,
                name="smoke-plan",
                amount=99900,
                currency="INR",
                interval="monthly",
                interval_count=1,
            )
        )
    )
    monkeypatch.setattr(
        subscription_service_module,
        "_subscription_to_response",
        lambda *args, **kwargs: SimpleNamespace(
            status=finalized.status, remote_sync=finalized.remote_sync
        ),
    )
    return service, repo


async def test_dead_create_row_finalizes_locally(monkeypatch: pytest.MonkeyPatch) -> None:
    row = _row(status="created", remote_sync="permanent_failure", provider_id=None)
    finalized = _row(
        status="cancelled", remote_sync="permanent_failure", provider_id=None
    )
    service, repo = _make_service(monkeypatch, row=row, finalized=finalized)

    result = await service.cancel_subscription(
        row.user_id, str(row.id), SubscriptionCancelDTO(cancel_at_period_end=False)
    )

    assert result.unwrap().status == "cancelled"
    repo.update_status.assert_awaited_once()
    call = repo.update_status.await_args
    new_status = call.kwargs.get("new_status") or call.args[1]
    assert new_status.value == "cancelled"


async def test_pending_create_still_refuses_terminalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _row(status="created", remote_sync="pending", provider_id=None)
    service, repo = _make_service(
        monkeypatch, row=row, finalized=_row(status="cancelled", remote_sync="pending", provider_id=None)
    )

    result = await service.cancel_subscription(
        row.user_id, str(row.id), SubscriptionCancelDTO(cancel_at_period_end=False)
    )

    assert isinstance(result.failure(), SubscriptionInvalidTransitionError)
    repo.update_status.assert_not_awaited()
