"""Credit feature errors and closed Result contract."""

from enum import StrEnum
from typing import ClassVar, assert_never

from returns.result import Result

from app.shared.result.errors import ErrorKind, FeatureError, http_status_for_kind


class CreditCode(StrEnum):
    CONFLICT = "CREDIT_CONFLICT"
    CONSUMPTION_CONFLICT = "CONSUMPTION_CONFLICT"
    NOT_FOUND = "CREDIT_NOT_FOUND"
    AMOUNT_MUST_BE_POSITIVE = "CREDIT_AMOUNT_MUST_BE_POSITIVE"
    METADATA_MISSING = "CREDIT_METADATA_MISSING"
    INVALID = "CREDIT_INVALID"
    DATABASE_ERROR = "CREDIT_DATABASE_ERROR"
    COLLABORATOR_ERROR = "CREDIT_COLLABORATOR_ERROR"


class CreditConflictError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.CONFLICT
    code: ClassVar[CreditCode] = CreditCode.CONFLICT
    retryable: ClassVar[bool] = False


class CreditConsumptionConflictError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.CONFLICT
    code: ClassVar[CreditCode] = CreditCode.CONSUMPTION_CONFLICT
    retryable: ClassVar[bool] = False


class CreditNotFoundError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.NOT_FOUND
    code: ClassVar[CreditCode] = CreditCode.NOT_FOUND
    retryable: ClassVar[bool] = False


class CreditAmountError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.VALIDATION
    code: ClassVar[CreditCode] = CreditCode.AMOUNT_MUST_BE_POSITIVE
    retryable: ClassVar[bool] = False


class CreditMetadataError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.VALIDATION
    code: ClassVar[CreditCode] = CreditCode.METADATA_MISSING
    retryable: ClassVar[bool] = False


class CreditValidationError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.VALIDATION
    code: ClassVar[CreditCode] = CreditCode.INVALID
    retryable: ClassVar[bool] = False


class CreditInfrastructureError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.INFRASTRUCTURE
    code: ClassVar[CreditCode] = CreditCode.DATABASE_ERROR
    retryable: ClassVar[bool] = False


class CreditCollaboratorError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.INFRASTRUCTURE
    code: ClassVar[CreditCode] = CreditCode.COLLABORATOR_ERROR
    retryable: ClassVar[bool] = False


type CreditError = (
    CreditConflictError
    | CreditConsumptionConflictError
    | CreditNotFoundError
    | CreditAmountError
    | CreditMetadataError
    | CreditValidationError
    | CreditInfrastructureError
    | CreditCollaboratorError
)
type CreditResult[T] = Result[T, CreditError]


def credit_error_to_http_status(error: CreditError) -> int:
    match error:
        case CreditConflictError() | CreditConsumptionConflictError():
            return http_status_for_kind(error.kind)
        case CreditNotFoundError():
            return http_status_for_kind(error.kind)
        case CreditAmountError() | CreditMetadataError() | CreditValidationError():
            return http_status_for_kind(error.kind)
        case CreditInfrastructureError() | CreditCollaboratorError():
            return http_status_for_kind(error.kind, retryable=error.retryable)
        case _ as unreachable:
            assert_never(unreachable)
