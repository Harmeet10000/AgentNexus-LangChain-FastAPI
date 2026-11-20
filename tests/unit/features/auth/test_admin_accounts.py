"""Behavioral coverage for the auth administrative account commands."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import Response
from returns.result import Failure, Success

from app.features.auth.application.commands.admin_accounts import AdminAccounts
from app.features.auth.domain.errors import AuthAdminConflictError, AuthAdminInfrastructureError
from app.features.auth.domain.user import UserRole
from app.shared.result.render import render_result


async def test_self_role_change_is_a_renderable_conflict() -> None:
    service = AdminAccounts(AsyncMock(), AsyncMock())
    result = await service.update_role("same", SimpleNamespace(), "same")

    assert isinstance(result, Failure)
    assert isinstance(result.failure(), AuthAdminConflictError)

    response = Response()
    envelope = render_result(result, response, message="User role updated")
    assert response.status_code == 409
    assert envelope.success is False


async def test_auth_session_failure_is_translated_to_admin_infrastructure() -> None:
    user = SimpleNamespace(id="target", is_active=True)
    user_repo = SimpleNamespace(
        find_by_id=AsyncMock(return_value=Success(user)),
        set_active=AsyncMock(return_value=Success(user)),
    )
    auth_error = SimpleNamespace(message="redis unavailable", details={"backend": "redis"})
    token_repo = SimpleNamespace(
        revoke_all_user_sessions=AsyncMock(return_value=Failure(auth_error))
    )
    service = AdminAccounts(user_repo, token_repo)

    result = await service.set_active("target", is_active=False, requesting_admin_id="admin")

    assert isinstance(result, Failure)
    assert isinstance(result.failure(), AuthAdminInfrastructureError)


def _user(*, role: UserRole = UserRole.USER, pending: bool = False) -> object:
    return type(
        "User",
        (),
        {
            "id": "target",
            "email": "target@example.com",
            "full_name": None,
            "role": role,
            "role_revision": 4 if pending else 0,
            "session_revocation_pending": pending,
            "is_verified": True,
            "is_active": True,
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
            "oauth_accounts": [],
        },
    )()


async def test_role_change_revokes_target_sessions() -> None:
    user = _user()
    user_repo = type(
        "UserRepo",
        (),
        {
            "find_by_id": AsyncMock(return_value=Success(user)),
            "update_role": AsyncMock(return_value=Success(user)),
        },
    )()
    token_repo = type(
        "TokenRepo",
        (),
        {"revoke_all_user_sessions": AsyncMock(return_value=Success([]))},
    )()

    result = await AdminAccounts(user_repo, token_repo).update_role(
        "target", UserRole.ADMIN, "admin-user"
    )

    assert isinstance(result, Success)
    token_repo.revoke_all_user_sessions.assert_awaited_once_with(
        user_id="target", reason="role_changed", authz_revision=0
    )


async def test_noop_role_change_does_not_revoke_sessions() -> None:
    user = _user()
    user_repo = type(
        "UserRepo",
        (),
        {
            "find_by_id": AsyncMock(return_value=Success(user)),
            "update_role": AsyncMock(return_value=Success(user)),
        },
    )()
    token_repo = type(
        "TokenRepo",
        (),
        {"revoke_all_user_sessions": AsyncMock(return_value=Success([]))},
    )()

    result = await AdminAccounts(user_repo, token_repo).update_role(
        "target", UserRole.USER, "admin-user"
    )

    assert isinstance(result, Success)
    user_repo.update_role.assert_not_awaited()
    token_repo.revoke_all_user_sessions.assert_not_awaited()


async def test_pending_role_revocation_retries_and_clears_marker() -> None:
    user = _user(role=UserRole.ADMIN, pending=True)
    clear_pending = AsyncMock(return_value=Success(user))
    user_repo = type(
        "UserRepo",
        (),
        {
            "find_by_id": AsyncMock(return_value=Success(user)),
            "clear_session_revocation_pending": clear_pending,
            "update_role": AsyncMock(),
        },
    )()
    token_repo = type(
        "TokenRepo",
        (),
        {"revoke_all_user_sessions": AsyncMock(return_value=Success(["session-1"]))},
    )()

    result = await AdminAccounts(user_repo, token_repo).update_role(
        "target", UserRole.ADMIN, "admin-user"
    )

    assert isinstance(result, Success)
    token_repo.revoke_all_user_sessions.assert_awaited_once_with(
        user_id="target", reason="role_changed", authz_revision=4
    )
    clear_pending.assert_awaited_once_with(user, expected_revision=4)
