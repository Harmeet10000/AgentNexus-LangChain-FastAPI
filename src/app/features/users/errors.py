"""Users feature typed errors."""

from enum import StrEnum
from typing import ClassVar, assert_never

from returns.result import Result

from app.shared.result import ErrorKind, FeatureError, http_status_for_kind


class UsersCode(StrEnum):
    INVALID_USER_ID = "INVALID_USER_ID"
    USER_NOT_FOUND = "USER_NOT_FOUND"
    SELF_ROLE_CHANGE = "SELF_ROLE_CHANGE"
    SELF_DEACTIVATION = "SELF_DEACTIVATION"
    SELF_DELETION = "SELF_DELETION"
    SELF_IMPERSONATION = "SELF_IMPERSONATION"
    IMPERSONATION_FORBIDDEN = "IMPERSONATION_FORBIDDEN"
    PERSISTENCE_ERROR = "USERS_PERSISTENCE_ERROR"
    SESSION_REVOCATION_ERROR = "SESSION_REVOCATION_ERROR"


class UsersValidationError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.VALIDATION
    code: ClassVar[UsersCode] = UsersCode.INVALID_USER_ID
    retryable: ClassVar[bool] = False


class UsersNotFoundError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.NOT_FOUND
    code: ClassVar[UsersCode] = UsersCode.USER_NOT_FOUND
    retryable: ClassVar[bool] = False

    user_id: str


class UsersConflictError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.CONFLICT
    code: ClassVar[UsersCode] = UsersCode.SELF_ROLE_CHANGE
    retryable: ClassVar[bool] = False

    operation: str


class UsersAuthorizationError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.AUTHORIZATION
    code: ClassVar[UsersCode] = UsersCode.IMPERSONATION_FORBIDDEN
    retryable: ClassVar[bool] = False

    operation: str


class UsersInfrastructureError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.INFRASTRUCTURE
    code: ClassVar[UsersCode] = UsersCode.PERSISTENCE_ERROR
    retryable: ClassVar[bool] = True

    operation: str


type UsersError = (
    UsersValidationError
    | UsersNotFoundError
    | UsersConflictError
    | UsersAuthorizationError
    | UsersInfrastructureError
)
type UsersResult[T] = Result[T, UsersError]


def users_error_to_http_status(error: UsersError) -> int:
    match error:
        case UsersValidationError() | UsersNotFoundError() | UsersConflictError():
            return http_status_for_kind(error.kind)
        case UsersAuthorizationError():
            return http_status_for_kind(error.kind)
        case UsersInfrastructureError():
            return http_status_for_kind(error.kind, retryable=error.retryable)
        case _ as unreachable:
            assert_never(unreachable)
