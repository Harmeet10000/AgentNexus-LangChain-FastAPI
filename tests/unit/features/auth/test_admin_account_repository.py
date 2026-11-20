"""Identity ownership contract (restructure-domain-boundaries task 5.3).

The administrative auth adapter performs no driver I/O for plain document
operations: every read/write goes through auth's identity contract. These tests
pin the delegation path using a stub document so no database is required.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from returns.result import Failure, Success

if TYPE_CHECKING:
    from typing import Any

from app.features.auth.adapters.persistence.mongodb import admin_accounts
from app.features.auth.adapters.persistence.mongodb.admin_accounts import AdminAccountRepository
from app.features.auth.domain.errors import (
    AuthAdminInfrastructureError,
    AuthInfrastructureError,
    AuthNotFoundError,
    AuthValidationError,
)
from app.features.auth.domain.user import UserRole

pytestmark = pytest.mark.unit


def _stub_user() -> Any:
    return SimpleNamespace(id="507f1f77bcf86cd799439011", role=UserRole.USER, is_active=True)


async def test_find_by_id_delegates_to_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    async def _recorded(user_id: str) -> Any:
        seen.append(user_id)
        return Success(_stub_user())

    monkeypatch.setattr(admin_accounts, "get_user_document", _recorded)

    result = await AdminAccountRepository.find_by_id("507f1f77bcf86cd799439011")

    assert seen == ["507f1f77bcf86cd799439011"]
    assert isinstance(result.unwrap(), SimpleNamespace)


async def test_find_by_id_translates_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _invalid(_user_id: str) -> Any:
        return Failure(
            AuthValidationError(
                message="Invalid user identifier",
                details={"user_id": "nope"},
                source="auth_identity",
            )
        )

    monkeypatch.setattr(admin_accounts, "get_user_document", _invalid)

    result = await AdminAccountRepository.find_by_id("nope")

    assert isinstance(result, Failure)
    error = result.failure()
    assert isinstance(error, AuthValidationError)
    assert error.details is not None
    assert error.details["user_id"] == "nope"


async def test_find_by_id_translates_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _missing(_user_id: str) -> Any:
        return Failure(
            AuthNotFoundError(
                message="gone",
                details={},
                source="auth_identity",
            )
        )

    monkeypatch.setattr(admin_accounts, "get_user_document", _missing)

    result = await AdminAccountRepository.find_by_id("507f1f77bcf86cd799439011")

    assert isinstance(result, Failure)
    error = result.failure()
    assert isinstance(error, AuthNotFoundError)
    assert error.details == {"user_id": "507f1f77bcf86cd799439011"}


async def test_update_role_delegates_and_translates_infra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[Any, Any]] = []

    async def _recorded(user: Any, role: Any) -> Any:
        seen.append((user, role))
        return Failure(
            AuthInfrastructureError(
                message="User role update failed",
                details={"user_id": "x", "error": "boom"},
                source="auth_identity",
            )
        )

    monkeypatch.setattr(admin_accounts, "set_user_role", _recorded)
    user = _stub_user()

    result = await AdminAccountRepository.update_role(user, UserRole.ADMIN)

    assert seen == [(user, UserRole.ADMIN)]
    assert isinstance(result, Failure)
    error = result.failure()
    assert isinstance(error, AuthAdminInfrastructureError)


async def test_hard_delete_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Any] = []

    async def _recorded(user: Any) -> Any:
        seen.append(user)
        return Success(None)

    monkeypatch.setattr(admin_accounts, "delete_user_document", _recorded)
    user = _stub_user()

    result = await AdminAccountRepository.hard_delete(user)

    assert seen == [user]
    assert result.unwrap() is None
