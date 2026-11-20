"""Authentication feature typed errors."""

from enum import StrEnum
from typing import ClassVar, assert_never, cast

from pydantic import model_validator
from returns.result import Result

from app.kernel.errors import ErrorKind, FeatureError
from app.shared.result.render import http_status_for_kind


class AuthCode(StrEnum):
    INVALID_USER_ID = "INVALID_USER_ID"
    USER_NOT_FOUND = "USER_NOT_FOUND"
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    USER_CONFLICT = "USER_CONFLICT"
    INVALID_CREDENTIALS = "INVALID_CREDENTIALS"
    REFRESH_TOKEN_REUSE = "REFRESH_TOKEN_REUSE"  # noqa: S105 — typed error code, not a secret
    FORBIDDEN = "FORBIDDEN"
    DATABASE_ERROR = "DATABASE_ERROR"
    # Correct credentials for an account whose authoritative SQL tenant/membership
    # does not exist yet.  Retryable by design: never report invalid credentials.
    ACCOUNT_PROVISIONING_PENDING = "ACCOUNT_PROVISIONING_PENDING"


class AuthValidationError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.VALIDATION
    code: ClassVar[AuthCode] = AuthCode.INVALID_USER_ID
    retryable: ClassVar[bool] = False


class AuthNotFoundError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.NOT_FOUND
    code: ClassVar[AuthCode] = AuthCode.USER_NOT_FOUND
    retryable: ClassVar[bool] = False


class AuthSessionNotFoundError(FeatureError):
    """A session-scoped 404: the addressed session does not exist (or is not
    the caller's).  Renders as ``SESSION_NOT_FOUND`` — never ``USER_NOT_FOUND``,
    which mislabeled session revocation failures in the 2026-11-15 sweep."""

    kind: ClassVar[ErrorKind] = ErrorKind.NOT_FOUND
    code: ClassVar[AuthCode] = AuthCode.SESSION_NOT_FOUND
    retryable: ClassVar[bool] = False

    session_id: str | None = None


class AuthConflictError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.CONFLICT
    code: ClassVar[AuthCode] = AuthCode.USER_CONFLICT
    retryable: ClassVar[bool] = False


class AuthAuthenticationError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.AUTHENTICATION
    code: ClassVar[AuthCode] = AuthCode.INVALID_CREDENTIALS
    retryable: ClassVar[bool] = False


class AuthRefreshReuseError(FeatureError):
    """A superseded refresh token was presented — rotation reuse detected.

    The whole token family (and its session) is already revoked by the
    rotation protocol before this failure is returned; the client must
    re-authenticate. Renders as 401 with ``REFRESH_TOKEN_REUSE``.
    """

    kind: ClassVar[ErrorKind] = ErrorKind.AUTHENTICATION
    code: ClassVar[AuthCode] = AuthCode.REFRESH_TOKEN_REUSE
    retryable: ClassVar[bool] = False


class AuthAuthorizationError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.AUTHORIZATION
    code: ClassVar[AuthCode] = AuthCode.FORBIDDEN
    retryable: ClassVar[bool] = False


class AuthInfrastructureError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.INFRASTRUCTURE
    code: ClassVar[AuthCode] = AuthCode.DATABASE_ERROR
    retryable: ClassVar[bool] = True

    @model_validator(mode="before")
    @classmethod
    def _sanitize_details(cls, value: object) -> object:
        """Keep driver and credential material out of HTTP error envelopes."""
        if not isinstance(value, dict):
            return value
        details = value.get("details")
        details_dict = cast("dict[str, object]", details) if isinstance(details, dict) else {}
        operation = details_dict.get("operation")
        safe_details = {"operation": operation} if isinstance(operation, str) else None
        return {**value, "details": safe_details}


class AuthProvisioningPendingError(FeatureError):
    """Correct credentials, but the authoritative SQL membership is missing.

    This is a deliberate, retryable state — never an authentication failure.
    Registration issues a session only after SQL provisioning commits, so this
    error surfaces when a durable provisioning step has not converged (for
    example a legacy account the migration has not repaired yet, or SQL state
    that was lost).  Renders as 409 with ``ACCOUNT_PROVISIONING_PENDING``.
    """

    kind: ClassVar[ErrorKind] = ErrorKind.CONFLICT
    code: ClassVar[AuthCode] = AuthCode.ACCOUNT_PROVISIONING_PENDING
    retryable: ClassVar[bool] = True


type AuthError = (
    AuthValidationError
    | AuthNotFoundError
    | AuthSessionNotFoundError
    | AuthConflictError
    | AuthAuthenticationError
    | AuthRefreshReuseError
    | AuthAuthorizationError
    | AuthInfrastructureError
    | AuthProvisioningPendingError
)
type AuthResult[T] = Result[T, AuthError]


def auth_error_to_http_status(error: AuthError) -> int:
    match error:
        case (
            AuthValidationError()
            | AuthNotFoundError()
            | AuthSessionNotFoundError()
            | AuthConflictError()
        ):
            return http_status_for_kind(error.kind)
        case (
            AuthAuthenticationError()
            | AuthRefreshReuseError()
            | AuthAuthorizationError()
            | AuthProvisioningPendingError()
        ):
            return http_status_for_kind(error.kind)
        case AuthInfrastructureError():
            return http_status_for_kind(error.kind, retryable=error.retryable)
        case _ as unreachable:
            assert_never(unreachable)
