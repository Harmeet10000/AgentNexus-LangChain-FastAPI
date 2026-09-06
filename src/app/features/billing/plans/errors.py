"""Plans feature typed errors."""

from enum import StrEnum
from typing import ClassVar, assert_never

from returns.result import Result

from app.shared.result import ErrorKind, FeatureError, http_status_for_kind


class PlanCode(StrEnum):
    NOT_FOUND = "PLAN_NOT_FOUND"
    DUPLICATE = "DUPLICATE_PLAN"
    INVALID_UPDATE = "INVALID_PLAN_UPDATE"
    DATABASE_ERROR = "PLAN_DATABASE_ERROR"


class PlanNotFoundError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.NOT_FOUND
    code: ClassVar[PlanCode] = PlanCode.NOT_FOUND
    retryable: ClassVar[bool] = False

    plan_id: str


class PlanConflictError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.CONFLICT
    code: ClassVar[PlanCode] = PlanCode.DUPLICATE
    retryable: ClassVar[bool] = False


class PlanValidationError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.VALIDATION
    code: ClassVar[PlanCode] = PlanCode.INVALID_UPDATE
    retryable: ClassVar[bool] = False


class PlanInfrastructureError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.INFRASTRUCTURE
    code: ClassVar[PlanCode] = PlanCode.DATABASE_ERROR
    retryable: ClassVar[bool] = False

    operation: str


type PlanError = (
    PlanNotFoundError | PlanConflictError | PlanValidationError | PlanInfrastructureError
)
type PlanResult[T] = Result[T, PlanError]


def plan_error_to_http_status(error: PlanError) -> int:
    match error:
        case PlanNotFoundError() | PlanConflictError() | PlanValidationError():
            return http_status_for_kind(error.kind)
        case PlanInfrastructureError():
            return http_status_for_kind(error.kind, retryable=error.retryable)
        case _ as unreachable:
            assert_never(unreachable)
