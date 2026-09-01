"""Authentication feature typed errors."""

from enum import StrEnum
from typing import ClassVar, assert_never

from returns.result import Result

from app.shared.result import ErrorKind, FeatureError, http_status_for_kind


class AuthCode(StrEnum):
    INVALID_USER_ID = "INVALID_USER_ID"
    USER_NOT_FOUND = "USER_NOT_FOUND"
    USER_CONFLICT = "USER_CONFLICT"
    INVALID_CREDENTIALS = "INVALID_CREDENTIALS"
    FORBIDDEN = "FORBIDDEN"
    DATABASE_ERROR = "DATABASE_ERROR"


class AuthValidationError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.VALIDATION
    code: ClassVar[AuthCode] = AuthCode.INVALID_USER_ID
    retryable: ClassVar[bool] = False


class AuthNotFoundError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.NOT_FOUND
    code: ClassVar[AuthCode] = AuthCode.USER_NOT_FOUND
    retryable: ClassVar[bool] = False


class AuthConflictError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.CONFLICT
    code: ClassVar[AuthCode] = AuthCode.USER_CONFLICT
    retryable: ClassVar[bool] = False


class AuthAuthenticationError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.AUTHENTICATION
    code: ClassVar[AuthCode] = AuthCode.INVALID_CREDENTIALS
    retryable: ClassVar[bool] = False


class AuthAuthorizationError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.AUTHORIZATION
    code: ClassVar[AuthCode] = AuthCode.FORBIDDEN
    retryable: ClassVar[bool] = False


class AuthInfrastructureError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.INFRASTRUCTURE
    code: ClassVar[AuthCode] = AuthCode.DATABASE_ERROR
    retryable: ClassVar[bool] = True


type AuthError = (
    AuthValidationError
    | AuthNotFoundError
    | AuthConflictError
    | AuthAuthenticationError
    | AuthAuthorizationError
    | AuthInfrastructureError
)
type AuthResult[T] = Result[T, AuthError]


def auth_error_to_http_status(error: AuthError) -> int:
    match error:
        case AuthValidationError() | AuthNotFoundError() | AuthConflictError():
            return http_status_for_kind(error.kind)
        case AuthAuthenticationError() | AuthAuthorizationError():
            return http_status_for_kind(error.kind)
        case AuthInfrastructureError():
            return http_status_for_kind(error.kind, retryable=error.retryable)
        case _ as unreachable:
            assert_never(unreachable)
