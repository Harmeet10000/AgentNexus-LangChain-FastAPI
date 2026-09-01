"""Closed error contracts for shared third-party services."""

from enum import StrEnum
from typing import ClassVar

from returns.result import Result

from app.shared.result import ErrorKind, FeatureError


class StorageCode(StrEnum):
    INVALID_INPUT = "STORAGE_INVALID_INPUT"
    UNAVAILABLE = "STORAGE_UNAVAILABLE"


class StorageValidationError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.VALIDATION
    code: ClassVar[StorageCode] = StorageCode.INVALID_INPUT


class StorageUnavailableError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.INFRASTRUCTURE
    code: ClassVar[StorageCode] = StorageCode.UNAVAILABLE
    retryable: ClassVar[bool] = True
    operation: str


type StorageError = StorageValidationError | StorageUnavailableError
type StorageResult[T] = Result[T, StorageError]


class TavilyCode(StrEnum):
    INVALID_INPUT = "TAVILY_INVALID_INPUT"
    REQUEST_FAILED = "TAVILY_REQUEST_FAILED"
    INVALID_RESPONSE = "TAVILY_INVALID_RESPONSE"


class TavilyValidationError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.VALIDATION
    code: ClassVar[TavilyCode] = TavilyCode.INVALID_INPUT


class TavilyExternalError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.EXTERNAL_SERVICE
    code: ClassVar[TavilyCode] = TavilyCode.REQUEST_FAILED
    retryable: ClassVar[bool] = True


class TavilyInvalidResponseError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.EXTERNAL_SERVICE
    code: ClassVar[TavilyCode] = TavilyCode.INVALID_RESPONSE


type TavilyError = TavilyValidationError | TavilyExternalError | TavilyInvalidResponseError
type TavilyResult[T] = Result[T, TavilyError]


class MailerCode(StrEnum):
    DELIVERY_FAILED = "MAILER_DELIVERY_FAILED"
    UNREACHABLE = "MAILER_UNREACHABLE"


class MailerDeliveryError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.EXTERNAL_SERVICE
    code: ClassVar[MailerCode] = MailerCode.DELIVERY_FAILED


class MailerUnavailableError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.EXTERNAL_SERVICE
    code: ClassVar[MailerCode] = MailerCode.UNREACHABLE
    retryable: ClassVar[bool] = True


type MailerError = MailerDeliveryError | MailerUnavailableError
type MailerResult[T] = Result[T, MailerError]
