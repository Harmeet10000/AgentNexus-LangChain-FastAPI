"""Invoice feature errors and closed Result contract."""

from enum import StrEnum
from typing import ClassVar, assert_never

from returns.result import Result

from app.shared.result.errors import ErrorKind, FeatureError, http_status_for_kind


class InvoiceCode(StrEnum):
    CONFLICT = "INVOICE_CONFLICT"
    NOT_FOUND = "INVOICE_NOT_FOUND"
    INVALID = "INVOICE_INVALID"
    DATABASE_ERROR = "INVOICE_DATABASE_ERROR"
    COLLABORATOR_ERROR = "INVOICE_COLLABORATOR_ERROR"
    STORAGE_ERROR = "INVOICE_STORAGE_ERROR"


class InvoiceConflictError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.CONFLICT
    code: ClassVar[InvoiceCode] = InvoiceCode.CONFLICT
    retryable: ClassVar[bool] = False


class InvoiceNotFoundError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.NOT_FOUND
    code: ClassVar[InvoiceCode] = InvoiceCode.NOT_FOUND
    retryable: ClassVar[bool] = False


class InvoiceValidationError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.VALIDATION
    code: ClassVar[InvoiceCode] = InvoiceCode.INVALID
    retryable: ClassVar[bool] = False


class InvoiceInfrastructureError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.INFRASTRUCTURE
    code: ClassVar[InvoiceCode] = InvoiceCode.DATABASE_ERROR
    retryable: ClassVar[bool] = False


class InvoiceCollaboratorError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.INFRASTRUCTURE
    code: ClassVar[InvoiceCode] = InvoiceCode.COLLABORATOR_ERROR
    retryable: ClassVar[bool] = False


class InvoiceStorageError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.EXTERNAL_SERVICE
    code: ClassVar[InvoiceCode] = InvoiceCode.STORAGE_ERROR
    retryable: ClassVar[bool] = True


type InvoiceError = (
    InvoiceConflictError
    | InvoiceNotFoundError
    | InvoiceValidationError
    | InvoiceInfrastructureError
    | InvoiceCollaboratorError
    | InvoiceStorageError
)
type InvoiceResult[T] = Result[T, InvoiceError]


def invoice_error_to_http_status(error: InvoiceError) -> int:
    match error:
        case InvoiceConflictError():
            return http_status_for_kind(error.kind)
        case InvoiceNotFoundError():
            return http_status_for_kind(error.kind)
        case InvoiceValidationError():
            return http_status_for_kind(error.kind)
        case InvoiceInfrastructureError():
            return http_status_for_kind(error.kind, retryable=error.retryable)
        case InvoiceCollaboratorError():
            return http_status_for_kind(error.kind, retryable=error.retryable)
        case InvoiceStorageError():
            return http_status_for_kind(error.kind)
        case _ as unreachable:
            assert_never(unreachable)
