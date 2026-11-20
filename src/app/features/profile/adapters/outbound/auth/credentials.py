"""Translate auth-owned credential use cases into profile-owned failures."""

from __future__ import annotations

from typing import TYPE_CHECKING

from returns.result import Failure, Success

from app.features.auth.public import PasswordChange, change_password
from app.kernel.errors import ErrorKind

from ....domain.errors import (
    ProfileAuthenticationError,
    ProfileConflictError,
    ProfileInfrastructureError,
)

if TYPE_CHECKING:
    from typing import Any

    from ....application.dto.profile import ChangePasswordCommand
    from ....domain.errors import ProfileError, ProfileResult


def _credential_failure(error: object) -> ProfileError:
    details = getattr(error, "details", None)
    message = getattr(error, "message", "Credential change failed")
    match getattr(error, "kind", ErrorKind.INFRASTRUCTURE):
        case ErrorKind.AUTHENTICATION:
            return ProfileAuthenticationError(message=message, details=details)
        case ErrorKind.CONFLICT:
            return ProfileConflictError(
                message=message,
                details=details,
                operation="change_password",
            )
        case _:
            return ProfileInfrastructureError(
                message=message,
                details=details,
                operation="change_password",
            )


class AuthCredentialGateway:
    """Call auth's public credential use case without exposing its adapters."""

    def __init__(self, account_store: Any, session_port: Any, password_hasher: Any) -> None:
        self._account_store = account_store
        self._session_port = session_port
        self._password_hasher = password_hasher

    async def change_password(self, command: ChangePasswordCommand) -> ProfileResult[None]:
        result = await change_password(
            self._account_store,
            self._session_port,
            self._password_hasher,
            command.profile.id,
            PasswordChange(
                current_password=command.current_password,
                new_password=command.new_password,
                revoke_other_sessions=command.revoke_other_sessions,
            ),
            current_session_id=command.current_session_id,
        )
        if isinstance(result, Failure):
            return Failure(_credential_failure(result.failure()))
        return Success(None)
