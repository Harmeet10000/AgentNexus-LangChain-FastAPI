"""Structural and behavioral tests for the profile slice."""

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from returns.result import Failure, Success

from app.features.profile.application.commands.update_profile import (
    UpdateProfileCommandHandler,
)
from app.features.profile.application.dto.profile import (
    ChangePasswordCommand,
    GetProfileQuery,
)
from app.features.profile.application.queries.get_profile import GetProfileQueryHandler
from app.features.profile.domain.entities.profile import Profile
from app.features.profile.domain.errors import ProfileAuthenticationError
from app.features.profile.module import MODULE

_PROFILE_ROOT = Path(__file__).parents[4] / "src/app/features/profile"


def _profile(*, avatar_url: str | None = None, password_hash: str | None = None) -> Profile:
    return Profile(
        id="507f1f77bcf86cd799439011",
        email="user@example.com",
        full_name="Ada Lovelace",
        password_hash=password_hash or "stored-hash",
        is_active=True,
        is_verified=True,
        role="user",
        avatar_url=avatar_url,
        created_at=datetime.now(UTC),
    )


def test_module_is_lazy_and_has_the_expected_identity() -> None:
    assert MODULE.name == "profile"
    assert len(MODULE.routes) == 1


def test_domain_and_application_are_transport_and_adapter_free() -> None:
    for root in (_PROFILE_ROOT / "domain", _PROFILE_ROOT / "application"):
        for path in root.rglob("*.py"):
            source = path.read_text()
            assert "fastapi" not in source
            assert "features.profile.adapters" not in source


async def test_invalid_password_is_returned_as_profile_authentication_error() -> None:
    handler = UpdateProfileCommandHandler(
        AsyncMock(),
        password_hasher=SimpleNamespace(verify=lambda _hash, _password: False),
        session_revoker=AsyncMock(),
        storage=None,
        logger=SimpleNamespace(),
    )

    result = await handler.change_password(
        ChangePasswordCommand(
            profile=_profile(),
            current_password="wrong",
            new_password="new",
            current_session_id=None,
            revoke_other_sessions=False,
        )
    )

    assert isinstance(result, Failure)
    assert isinstance(result.failure(), ProfileAuthenticationError)


async def test_query_resolves_private_avatar_through_the_storage_port() -> None:
    repository = SimpleNamespace(
        get_profile=AsyncMock(return_value=Success(_profile(avatar_url="s3://bucket/avatar")))
    )
    storage = SimpleNamespace(
        bucket="bucket",
        uri_to_location=lambda uri: Success(SimpleNamespace(bucket="bucket", key=uri.rsplit("/", 1)[1])),
        get_signed_get_url=AsyncMock(return_value=Success("https://signed.example/avatar")),
    )
    handler = GetProfileQueryHandler(repository, storage=storage)

    profile_result = await handler.get_profile(GetProfileQuery(user_id="user"))
    signed_result = await handler.signed_avatar_url(profile_result.unwrap().avatar_url)

    assert signed_result == Success("https://signed.example/avatar")
    storage.get_signed_get_url.assert_awaited_once_with(key="avatar")
