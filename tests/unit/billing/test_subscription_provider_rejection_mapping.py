"""G9b regression: a terminal provider rejection must render as a typed 4xx.

The 2026-11-15 sweep observed ``POST /subscriptions`` -> ``500 DATABASE_ERROR``
("The billing provider rejected the request", ``provider_code:
"invalid_request"``) while the local row persisted with
``remoteSync: "permanent_failure"`` and no provider id.  Two facts were wrong
with that response: the provider's INVALID_REQUEST verdict is client-facing
input (the request can never succeed), so it must render as a 4xx with the
provider code surfaced — and the persisted row's ``remote_sync:
"permanent_failure"`` is the billing state machine's pending-reconciliation
marker, which is truthful and retained.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from app.features.billing.exceptions import (
    LifecycleCoordinatorProviderError,
    LifecycleCoordinatorTransientProviderError,
)
from app.features.billing.providers import ProviderErrorCode
from app.features.billing.subscriptions.exceptions import (
    SubscriptionInfrastructureError,
    SubscriptionTransientInfrastructureError,
    SubscriptionValidationError,
    subscription_error_to_http_status,
)
from app.features.billing.subscriptions.service import SubscriptionService
from app.kernel.errors import ErrorKind

if TYPE_CHECKING:
    from app.features.billing.subscriptions.exceptions import SubscriptionError

pytestmark = pytest.mark.unit


def _provider_error(
    code: ProviderErrorCode, *, retryable: bool = False
) -> LifecycleCoordinatorProviderError | LifecycleCoordinatorTransientProviderError:
    cls = (
        LifecycleCoordinatorTransientProviderError
        if retryable
        else LifecycleCoordinatorProviderError
    )
    return cls(
        message="The billing provider rejected the request",
        details={"provider_code": str(code)},
        source="subscription_lifecycle_coordinator",
        provider_code=code,
        operation=None,
        correlation_id="corr-1",
        requires_reconciliation=False,
    )


def test_invalid_request_provider_rejection_maps_to_typed_4xx() -> None:
    error = SubscriptionService._translate_lifecycle_error(
        _provider_error(ProviderErrorCode.INVALID_REQUEST)
    )

    assert isinstance(error, SubscriptionValidationError)
    # 422 via the VALIDATION kind, and the provider verdict stays visible.
    assert subscription_error_to_http_status(error) == 422
    assert error.details is not None
    assert error.details["provider_code"] == str(ProviderErrorCode.INVALID_REQUEST)


def test_transient_provider_rejection_stays_infrastructure() -> None:
    """Retryable provider failures keep their retryable infrastructure shape."""
    error = SubscriptionService._translate_lifecycle_error(
        _provider_error(ProviderErrorCode.RATE_LIMITED, retryable=True)
    )

    assert isinstance(error, SubscriptionTransientInfrastructureError)
    assert error.retryable


def test_terminal_provider_rejection_without_invalid_request_stays_infrastructure() -> None:
    """A non-retryable provider failure that is not a request verdict is infra."""
    error = SubscriptionService._translate_lifecycle_error(
        _provider_error(ProviderErrorCode.AUTHENTICATION)
    )

    assert isinstance(error, SubscriptionInfrastructureError)
    assert not error.retryable


def test_invalid_request_mapping_keeps_the_error_in_the_closed_union() -> None:
    """The mapping is a union member, so the renderer contract stays intact."""
    error: SubscriptionError = SubscriptionService._translate_lifecycle_error(
        _provider_error(ProviderErrorCode.INVALID_REQUEST)
    )

    assert error.kind is ErrorKind.VALIDATION
    assert not error.retryable
