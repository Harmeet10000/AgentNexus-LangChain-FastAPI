"""Profile feature typed errors."""

from enum import StrEnum
from typing import ClassVar, assert_never

from returns.result import Result

from app.shared.result import ErrorKind, FeatureError, http_status_for_kind


class ProfileCode(StrEnum):
    OAUTH_CREDENTIAL_CHANGE = "OAUTH_CREDENTIAL_CHANGE"
    CURRENT_CREDENTIAL_INVALID = "CURRENT_CREDENTIAL_INVALID"
    CREDENTIAL_UNCHANGED = "CREDENTIAL_UNCHANGED"
    STORAGE_UNAVAILABLE = "PROFILE_STORAGE_UNAVAILABLE"
    AVATAR_UPLOAD_FAILED = "AVATAR_UPLOAD_FAILED"
    PERSISTENCE_ERROR = "PROFILE_PERSISTENCE_ERROR"
    SESSION_REVOCATION_ERROR = "PROFILE_SESSION_REVOCATION_ERROR"


class ProfileConflictError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.CONFLICT
    code: ClassVar[ProfileCode] = ProfileCode.CREDENTIAL_UNCHANGED
    retryable: ClassVar[bool] = False

    operation: str


class ProfileAuthenticationError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.AUTHENTICATION
    code: ClassVar[ProfileCode] = ProfileCode.CURRENT_CREDENTIAL_INVALID
    retryable: ClassVar[bool] = False


class ProfileValidationError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.VALIDATION
    code: ClassVar[ProfileCode] = ProfileCode.AVATAR_UPLOAD_FAILED


class ProfileInfrastructureError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.INFRASTRUCTURE
    code: ClassVar[ProfileCode] = ProfileCode.PERSISTENCE_ERROR
    retryable: ClassVar[bool] = True

    operation: str


class ProfileStorageError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.INFRASTRUCTURE
    code: ClassVar[ProfileCode] = ProfileCode.STORAGE_UNAVAILABLE
    retryable: ClassVar[bool] = True


type ProfileError = (
    ProfileConflictError
    | ProfileAuthenticationError
    | ProfileValidationError
    | ProfileInfrastructureError
    | ProfileStorageError
)
type ProfileResult[T] = Result[T, ProfileError]


def profile_error_to_http_status(error: ProfileError) -> int:
    match error:
        case ProfileConflictError() | ProfileAuthenticationError() | ProfileValidationError():
            return http_status_for_kind(error.kind)
        case ProfileInfrastructureError() | ProfileStorageError():
            return http_status_for_kind(error.kind, retryable=error.retryable)
        case _ as unreachable:
            assert_never(unreachable)
