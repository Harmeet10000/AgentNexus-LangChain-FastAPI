"""RAG provider-boundary typed errors."""

from enum import StrEnum
from typing import ClassVar

from returns.result import Result

from app.shared.result import ErrorKind, FeatureError


class RagCode(StrEnum):
    PROVIDER_FAILURE = "RAG_PROVIDER_FAILURE"
    INVALID_INPUT = "RAG_INVALID_INPUT"
    SERVICE_FAILURE = "RAG_SERVICE_FAILURE"


class RagProviderError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.EXTERNAL_SERVICE
    code: ClassVar[RagCode] = RagCode.PROVIDER_FAILURE
    retryable: ClassVar[bool] = True

    model: str
    text_count: int


class RagValidationError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.VALIDATION
    code: ClassVar[RagCode] = RagCode.INVALID_INPUT
    retryable: ClassVar[bool] = False


class RagServiceError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.EXTERNAL_SERVICE
    code: ClassVar[RagCode] = RagCode.SERVICE_FAILURE
    retryable: ClassVar[bool] = True

    operation: str


type RagError = RagProviderError | RagValidationError | RagServiceError
type RagResult[T] = Result[T, RagError]
