"""RAG provider-boundary typed errors."""

from enum import StrEnum
from typing import ClassVar

from returns.result import Result

from app.shared.result import ErrorKind, FeatureError


class RagCode(StrEnum):
    PROVIDER_FAILURE = "RAG_PROVIDER_FAILURE"


class RagProviderError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.EXTERNAL_SERVICE
    code: ClassVar[RagCode] = RagCode.PROVIDER_FAILURE
    retryable: ClassVar[bool] = True

    model: str
    text_count: int


type RagError = RagProviderError
type RagResult[T] = Result[T, RagError]
