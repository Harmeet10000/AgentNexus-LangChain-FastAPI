"""Payment feature errors and closed Result contract."""

from enum import StrEnum
from typing import ClassVar, assert_never

from returns.result import Result

from app.shared.result.errors import ErrorKind, FeatureError, http_status_for_kind


class PaymentCode(StrEnum):
    DUPLICATE = "DUPLICATE_PAYMENT"
    NOT_FOUND = "PAYMENT_NOT_FOUND"
    INVALID = "PAYMENT_INVALID"
    DATABASE_ERROR = "PAYMENT_DATABASE_ERROR"
    PROVIDER_ERROR = "PAYMENT_PROVIDER_ERROR"
    PROVIDER_UNAVAILABLE = "PAYMENT_PROVIDER_UNAVAILABLE"
    COLLABORATOR_ERROR = "PAYMENT_COLLABORATOR_ERROR"


class PaymentConflictError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.CONFLICT
    code: ClassVar[PaymentCode] = PaymentCode.DUPLICATE
    retryable: ClassVar[bool] = False


class PaymentNotFoundError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.NOT_FOUND
    code: ClassVar[PaymentCode] = PaymentCode.NOT_FOUND
    retryable: ClassVar[bool] = False


class PaymentValidationError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.VALIDATION
    code: ClassVar[PaymentCode] = PaymentCode.INVALID
    retryable: ClassVar[bool] = False


class PaymentInfrastructureError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.INFRASTRUCTURE
    code: ClassVar[PaymentCode] = PaymentCode.DATABASE_ERROR
    retryable: ClassVar[bool] = False


class PaymentProviderError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.EXTERNAL_SERVICE
    code: ClassVar[PaymentCode] = PaymentCode.PROVIDER_ERROR
    retryable: ClassVar[bool] = False


class PaymentProviderUnavailableError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.EXTERNAL_SERVICE
    code: ClassVar[PaymentCode] = PaymentCode.PROVIDER_UNAVAILABLE
    retryable: ClassVar[bool] = True


class PaymentCollaboratorError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.INFRASTRUCTURE
    code: ClassVar[PaymentCode] = PaymentCode.COLLABORATOR_ERROR
    retryable: ClassVar[bool] = False


type PaymentError = (
    PaymentConflictError
    | PaymentNotFoundError
    | PaymentValidationError
    | PaymentInfrastructureError
    | PaymentProviderError
    | PaymentProviderUnavailableError
    | PaymentCollaboratorError
)
type PaymentResult[T] = Result[T, PaymentError]


def payment_error_to_http_status(error: PaymentError) -> int:
    match error:
        case PaymentConflictError():
            return http_status_for_kind(error.kind)
        case PaymentNotFoundError():
            return http_status_for_kind(error.kind)
        case PaymentValidationError():
            return http_status_for_kind(error.kind)
        case PaymentInfrastructureError():
            return http_status_for_kind(error.kind, retryable=error.retryable)
        case PaymentProviderError() | PaymentProviderUnavailableError():
            return http_status_for_kind(error.kind)
        case PaymentCollaboratorError():
            return http_status_for_kind(error.kind, retryable=error.retryable)
        case _ as unreachable:
            assert_never(unreachable)
