"""Typed failures raised by audit application ports."""

from enum import StrEnum
from typing import ClassVar

from returns.result import Result

from app.kernel.errors import ErrorKind, FeatureError


class AuditCode(StrEnum):
    DATABASE_ERROR = "AUDIT_DATABASE_ERROR"


class AuditInfrastructureError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.INFRASTRUCTURE
    code: ClassVar[AuditCode] = AuditCode.DATABASE_ERROR
    retryable: ClassVar[bool] = False

    operation: str


type AuditError = AuditInfrastructureError
type AuditResult[T] = Result[T, AuditError]
