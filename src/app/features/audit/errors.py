"""Audit feature typed errors."""

from enum import StrEnum
from typing import ClassVar, assert_never

from returns.result import Result

from app.shared.result import ErrorKind, FeatureError, http_status_for_kind


class AuditCode(StrEnum):
    DATABASE_ERROR = "AUDIT_DATABASE_ERROR"


class AuditInfrastructureError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.INFRASTRUCTURE
    code: ClassVar[AuditCode] = AuditCode.DATABASE_ERROR
    retryable: ClassVar[bool] = False

    operation: str


type AuditError = AuditInfrastructureError
type AuditResult[T] = Result[T, AuditError]


def audit_error_to_http_status(error: AuditError) -> int:
    match error:
        case AuditInfrastructureError():
            return http_status_for_kind(error.kind, retryable=error.retryable)
        case _ as unreachable:
            assert_never(unreachable)
