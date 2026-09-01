"""Closed error contract for shared crawler processing."""

from enum import StrEnum
from typing import ClassVar

from returns.result import Result

from app.shared.result import ErrorKind, FeatureError


class CrawlerProcessingCode(StrEnum):
    UNKNOWN_SCHEMA = "UNKNOWN_EXTRACTION_SCHEMA"
    MISSING_SCHEMA = "MISSING_EXTRACTION_SCHEMA"
    INVALID_JSON = "INVALID_EXTRACTION_JSON"
    INVALID_SHAPE = "INVALID_EXTRACTION_SHAPE"
    CRAWL_FAILED = "CRAWL_FAILED"


class CrawlerProcessingValidationError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.VALIDATION
    code: ClassVar[CrawlerProcessingCode] = CrawlerProcessingCode.INVALID_SHAPE


class CrawlerProviderError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.EXTERNAL_SERVICE
    code: ClassVar[CrawlerProcessingCode] = CrawlerProcessingCode.CRAWL_FAILED
    retryable: ClassVar[bool] = True
    url: str


type CrawlerProcessingError = CrawlerProcessingValidationError | CrawlerProviderError
type CrawlerProcessingResult[T] = Result[T, CrawlerProcessingError]
