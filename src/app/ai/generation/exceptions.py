"""Typed, sanitized failures for the text-generation seam."""

from enum import StrEnum
from typing import ClassVar, Literal

from app.kernel.errors import ErrorKind, FeatureError


class TextGenerationCode(StrEnum):
    UNAVAILABLE = "TEXT_GENERATION_UNAVAILABLE"
    REJECTED = "TEXT_GENERATION_REJECTED"
    PROTOCOL_ERROR = "TEXT_GENERATION_PROTOCOL_ERROR"


class TextGenerationUnavailableError(FeatureError):
    """The provider cannot serve this request right now."""

    # Fixed public fields prevent adapter implementations from leaking provider
    # URLs, request content, credentials, or driver exception text.
    message: Literal["Text generation provider is unavailable"] = (
        "Text generation provider is unavailable"
    )
    details: None = None
    source: Literal["text_generation"] = "text_generation"

    kind: ClassVar[ErrorKind] = ErrorKind.EXTERNAL_SERVICE
    code: ClassVar[TextGenerationCode] = TextGenerationCode.UNAVAILABLE
    retryable: ClassVar[bool] = True


class TextGenerationRejectedError(FeatureError):
    """The provider rejected the generation request."""

    message: Literal["Text generation request was rejected"] = (
        "Text generation request was rejected"
    )
    details: None = None
    source: Literal["text_generation"] = "text_generation"

    kind: ClassVar[ErrorKind] = ErrorKind.EXTERNAL_SERVICE
    code: ClassVar[TextGenerationCode] = TextGenerationCode.REJECTED
    retryable: ClassVar[bool] = False


class TextGenerationProtocolError(FeatureError):
    """The provider returned a response outside its agreed wire contract."""

    message: Literal["Text generation provider returned an invalid response"] = (
        "Text generation provider returned an invalid response"
    )
    details: None = None
    source: Literal["text_generation"] = "text_generation"

    kind: ClassVar[ErrorKind] = ErrorKind.EXTERNAL_SERVICE
    code: ClassVar[TextGenerationCode] = TextGenerationCode.PROTOCOL_ERROR
    retryable: ClassVar[bool] = False


type TextGenerationError = (
    TextGenerationUnavailableError | TextGenerationRejectedError | TextGenerationProtocolError
)
