"""Identity ownership contract (restructure-domain-boundaries task 5.3).

The users feature performs no driver I/O for plain document operations: every
read/write goes through auth's identity contract, and identity failures arrive
as the users error union. These tests pin the delegation path and the
translation, using a stub document so no database is required.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from returns.result import Failure, Success

if TYPE_CHECKING:
    from typing import Any

from app.features.auth.exceptions import (
    AuthInfrastructureError,
    AuthNotFoundError,
    AuthValidationError,
)
from app.features.auth.models import UserRole
from app.features.users import repository as users_repository
from app.features.users.exceptions import (
    UsersInfrastructureError,
    UsersNotFoundError,
    UsersValidationError,
)
from app.features.users.repository import UserAdminRepository

pytestmark = pytest.mark.unit


def _stub_user() -> Any:
    return SimpleNamespace(id="507f1f77bcf86cd799439011", role=UserRole.USER, is_active=True)


async def test_find_by_id_delegates_to_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    async def _recorded(user_id: str) -> Any:
        seen.append(user_id)
        return Success(_stub_user())

    monkeypatch.setattr(users_repository, "get_user_document", _recorded)

    result = await UserAdminRepository.find_by_id("507f1f77bcf86cd799439011")

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

    monkeypatch.setattr(users_repository, "get_user_document", _invalid)

    result = await UserAdminRepository.find_by_id("nope")

    assert isinstance(result, Failure)
    error = result.failure()
    assert isinstance(error, UsersValidationError)
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

    monkeypatch.setattr(users_repository, "get_user_document", _missing)

    result = await UserAdminRepository.find_by_id("507f1f77bcf86cd799439011")

    assert isinstance(result, Failure)
    error = result.failure()
    assert isinstance(error, UsersNotFoundError)
    assert error.user_id == "507f1f77bcf86cd799439011"


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

    monkeypatch.setattr(users_repository, "set_user_role", _recorded)
    user = _stub_user()

    result = await UserAdminRepository.update_role(user, UserRole.ADMIN)

    assert seen == [(user, UserRole.ADMIN)]
    assert isinstance(result, Failure)
    error = result.failure()
    assert isinstance(error, UsersInfrastructureError)
    assert error.operation == "update_role"


async def test_hard_delete_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Any] = []

    async def _recorded(user: Any) -> Any:
        seen.append(user)
        return Success(None)

    monkeypatch.setattr(users_repository, "delete_user_document", _recorded)
    user = _stub_user()

    result = await UserAdminRepository.hard_delete(user)

    assert seen == [user]
    assert result.unwrap() is None
