"""Subscription infrastructure errors preserve permanent versus transient semantics."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import Response
from returns.result import Failure
from sqlalchemy.exc import SQLAlchemyError

from app.features.invoices.errors import InvoiceCollaboratorError
from app.features.invoices.service import _translate_collaborator_error
from app.features.plans.errors import PlanInfrastructureError
from app.features.subscriptions.errors import (
    SubscriptionInfrastructureError,
    SubscriptionTransientInfrastructureError,
)
from app.features.subscriptions.repository import SubscriptionRepository
from app.features.subscriptions.service import (
    SubscriptionService,
    subscription_error_to_http_status,
)
from app.shared.result import render_result


def _permanent_error() -> SubscriptionInfrastructureError:
    return SubscriptionInfrastructureError(
        message="permanent database failure",
        source="test",
        operation="test",
    )


def _transient_error() -> SubscriptionTransientInfrastructureError:
    return SubscriptionTransientInfrastructureError(
        message="transient database failure",
        source="test",
        operation="test",
    )


def test_infrastructure_retryability_is_fixed_by_concrete_type() -> None:
    permanent = _permanent_error()
    transient = _transient_error()

    assert permanent.retryable is False
    assert transient.retryable is True
    assert "retryable" not in permanent.__class__.model_fields
    assert "retryable" not in transient.__class__.model_fields


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (_permanent_error(), 500),
        (_transient_error(), 503),
    ],
)
def test_subscription_match_and_renderer_preserve_retryability(
    error: SubscriptionInfrastructureError | SubscriptionTransientInfrastructureError,
    expected_status: int,
) -> None:
    assert subscription_error_to_http_status(error) == expected_status

    response = Response()
    envelope = render_result(Failure(error), response)

    assert response.status_code == expected_status
    assert envelope.status_code == expected_status


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("retryable", "expected_type"),
    [
        (False, SubscriptionInfrastructureError),
        (True, SubscriptionTransientInfrastructureError),
    ],
)
async def test_plan_translation_selects_infrastructure_sibling(
    retryable: bool,
    expected_type: type[SubscriptionInfrastructureError | SubscriptionTransientInfrastructureError],
) -> None:
    plans = SimpleNamespace(
        find_by_id=AsyncMock(
            return_value=Failure(
                PlanInfrastructureError(
                    message="plan database failure",
                    source="plan_repository",
                    operation="find_by_id",
                )
            )
        )
    )
    service = object.__new__(SubscriptionService)
    service.plans = plans

    result = await service._load_plan("plan-id")

    assert isinstance(result, Failure)
    assert isinstance(result.failure(), SubscriptionInfrastructureError)
    assert result.failure().retryable is False


@pytest.mark.asyncio
async def test_repository_database_failure_is_transient() -> None:
    session = SimpleNamespace(
        execute=AsyncMock(side_effect=SQLAlchemyError("database unavailable")),
        rollback=AsyncMock(),
    )
    repository = SubscriptionRepository(session)

    result = await repository.find_by_id("subscription-id")

    assert isinstance(result, Failure)
    assert isinstance(result.failure(), SubscriptionTransientInfrastructureError)
    assert result.failure().retryable is True
    session.rollback.assert_awaited_once()


@pytest.mark.parametrize("error", [_permanent_error(), _transient_error()])
def test_invoice_translates_collaborator_failure_to_its_own_union(
    error: SubscriptionInfrastructureError | SubscriptionTransientInfrastructureError,
) -> None:
    translated = _translate_collaborator_error(error)

    assert isinstance(translated, InvoiceCollaboratorError)
    assert translated.message == error.message
