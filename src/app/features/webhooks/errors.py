"""Webhook feature errors and closed Result contract."""

from enum import StrEnum
from typing import ClassVar, assert_never

from returns.result import Result

from app.shared.result.errors import ErrorKind, FeatureError, http_status_for_kind


class WebhookCode(StrEnum):
    DUPLICATE = "DUPLICATE_WEBHOOK_EVENT"
    NOT_FOUND = "WEBHOOK_EVENT_NOT_FOUND"
    VERIFICATION_FAILED = "WEBHOOK_VERIFICATION_FAILED"
    INVALID = "WEBHOOK_INVALID"
    DATABASE_ERROR = "WEBHOOK_DATABASE_ERROR"
    COLLABORATOR_ERROR = "WEBHOOK_COLLABORATOR_ERROR"


class WebhookConflictError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.CONFLICT
    code: ClassVar[WebhookCode] = WebhookCode.DUPLICATE
    retryable: ClassVar[bool] = False


class WebhookNotFoundError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.NOT_FOUND
    code: ClassVar[WebhookCode] = WebhookCode.NOT_FOUND
    retryable: ClassVar[bool] = False


class WebhookVerificationError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.AUTHENTICATION
    code: ClassVar[WebhookCode] = WebhookCode.VERIFICATION_FAILED
    retryable: ClassVar[bool] = False


class WebhookValidationError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.VALIDATION
    code: ClassVar[WebhookCode] = WebhookCode.INVALID
    retryable: ClassVar[bool] = False


class WebhookInfrastructureError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.INFRASTRUCTURE
    code: ClassVar[WebhookCode] = WebhookCode.DATABASE_ERROR
    retryable: ClassVar[bool] = False


class WebhookCollaboratorError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.INFRASTRUCTURE
    code: ClassVar[WebhookCode] = WebhookCode.COLLABORATOR_ERROR
    retryable: ClassVar[bool] = False


type WebhookError = (
    WebhookConflictError
    | WebhookNotFoundError
    | WebhookVerificationError
    | WebhookValidationError
    | WebhookInfrastructureError
    | WebhookCollaboratorError
)
type WebhookResult[T] = Result[T, WebhookError]


def webhook_error_to_http_status(error: WebhookError) -> int:
    match error:
        case WebhookConflictError():
            return http_status_for_kind(error.kind)
        case WebhookNotFoundError():
            return http_status_for_kind(error.kind)
        case WebhookVerificationError():
            return http_status_for_kind(error.kind)
        case WebhookValidationError():
            return http_status_for_kind(error.kind)
        case WebhookInfrastructureError() | WebhookCollaboratorError():
            return http_status_for_kind(error.kind, retryable=error.retryable)
        case _ as unreachable:
            assert_never(unreachable)
