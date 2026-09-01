"""Dunning feature typed errors."""

from enum import StrEnum
from typing import ClassVar, assert_never

from returns.result import Result

from app.shared.result import ErrorKind, FeatureError, http_status_for_kind


class DunningCode(StrEnum):
    QUERY_FAILED = "DUNNING_QUERY_FAILED"
    PLAN_LOOKUP_FAILED = "DUNNING_PLAN_LOOKUP_FAILED"
    CHARGE_FAILED = "DUNNING_CHARGE_FAILED"
    SUBSCRIPTION_UPDATE_FAILED = "DUNNING_SUBSCRIPTION_UPDATE_FAILED"
    AUDIT_FAILED = "DUNNING_AUDIT_FAILED"


class DunningInfrastructureError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.INFRASTRUCTURE
    code: ClassVar[DunningCode] = DunningCode.QUERY_FAILED
    retryable: ClassVar[bool] = True

    operation: str


class DunningExternalServiceError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.EXTERNAL_SERVICE
    code: ClassVar[DunningCode] = DunningCode.CHARGE_FAILED
    retryable: ClassVar[bool] = True

    operation: str


type DunningError = DunningInfrastructureError | DunningExternalServiceError
type DunningResult[T] = Result[T, DunningError]


def dunning_error_to_http_status(error: DunningError) -> int:
    match error:
        case DunningInfrastructureError() | DunningExternalServiceError():
            return http_status_for_kind(error.kind, retryable=error.retryable)
        case _ as unreachable:
            assert_never(unreachable)
