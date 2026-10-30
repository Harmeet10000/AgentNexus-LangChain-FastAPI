"""Authorization-boundary error mapping: outage vs denial vs pending.

Regression coverage for the retryability drop that turned retryable SQL
failures into 403/500 responses, plus the typed
``ACCOUNT_PROVISIONING_PENDING`` rendering.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from returns.result import Failure, Success

from app.features.auth.adapters.persistence.mongodb.user_store import AuthAccount
from app.features.auth.api.http.dependencies import (
    _raise_tenant_selection_failure,
    get_current_authorization_context,
    get_current_websocket_authorization_context,
)
from app.features.auth.domain.errors import (
    AuthInfrastructureError,
    AuthNotFoundError,
    AuthProvisioningPendingError,
    auth_error_to_http_status,
)
from app.features.auth.domain.user import UserRole
from app.security.tokens import TokenClaims
from app.utils import ConflictException, ForbiddenException, InfrastructureException

if TYPE_CHECKING:
    from typing import Any


def _claims() -> TokenClaims:
    return TokenClaims(
        sub="user-1",
        jti="jti-1",
        sid="session-1",
        role="user",
        permissions=[],
        token_type="access",
    )


def _account() -> AuthAccount:
    now = datetime.now(UTC)
    return AuthAccount(
        id="user-1",
        email="user-1@example.com",
        full_name=None,
        password_hash=None,
        is_active=True,
        is_verified=True,
        role=UserRole.USER,
        permissions=frozenset(),
        created_at=now,
        updated_at=now,
    )


class _Headers:
    def __init__(self) -> None:
        self._values: dict[str, list[str]] = {}

    def get(self, _name: str) -> str | None:
        return None

    def getlist(self, name: str) -> list[str]:
        return self._values.get(name, [])


class _QueryParams:
    def getlist(self, _name: str) -> list[str]:
        return []


class _Connection:
    def __init__(self) -> None:
        self.headers = _Headers()
        self.cookies: dict[str, str] = {}
        self.query_params = _QueryParams()
        self.scope: dict[str, object] = {}


class _AccountStoreStub:
    async def find_by_id(self, _account_id: str):
        return Success(_account())


def _membership_service(live: Any) -> object:
    class _Service:
        async def resolve_tenant(self, *, user_id: str, requested_tenant_id: str | None = None):
            return Success("personal:user-1")

        async def get_live_authorization(self, *, tenant_id: str, user_id: str):
            return live

    return _Service()


@pytest.mark.asyncio
async def test_http_authorization_maps_sql_outage_to_retryable_503() -> None:
    outage = Failure(
        AuthInfrastructureError(
            message="Tenant authorization read failed",
            details={"operation": "get_live_authorization"},
            source="test",
        )
    )

    with pytest.raises(InfrastructureException) as raised:
        await get_current_authorization_context(
            _claims(),
            _account(),
            _membership_service(outage),
            _Connection(),
        )

    assert raised.value.status_code == 503
    assert raised.value.error_code == "DATABASE_ERROR"
    data = raised.value.detail["data"]
    assert data is not None
    assert data.get("operation") == "get_live_authorization"


@pytest.mark.asyncio
async def test_websocket_authorization_maps_sql_outage_to_retryable_503() -> None:
    outage = Failure(
        AuthInfrastructureError(
            message="Tenant authorization read failed",
            details={"operation": "get_live_authorization"},
            source="test",
        )
    )

    with pytest.raises(InfrastructureException) as raised:
        await get_current_websocket_authorization_context(
            _Connection(),  # type: ignore[arg-type] — only header/query access is exercised
            _claims(),
            _AccountStoreStub(),
            _membership_service(outage),
        )

    assert raised.value.status_code == 503


def test_missing_membership_fails_closed_as_denial_not_outage() -> None:
    with pytest.raises(ForbiddenException):
        _raise_tenant_selection_failure(
            AuthNotFoundError(
                message="No tenant membership found",
                details={"user_id": "user-1"},
                source="test",
            )
        )


def test_pending_error_renders_as_retryable_409_with_typed_code() -> None:
    error = AuthProvisioningPendingError(
        message="Account provisioning is pending; retry shortly",
        details={"user_id": "user-1", "retryable": True},
        source="test",
    )

    assert auth_error_to_http_status(error) == 409
    assert error.code.value == "ACCOUNT_PROVISIONING_PENDING"
    assert error.retryable is True

    exc = ConflictException(detail=error.message, error_code=error.code, data=error.details)
    assert exc.status_code == 409
    assert exc.error_code == "ACCOUNT_PROVISIONING_PENDING"
