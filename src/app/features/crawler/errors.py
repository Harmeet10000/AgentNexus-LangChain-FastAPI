"""Crawler feature typed errors."""

from enum import StrEnum
from typing import ClassVar, assert_never

from returns.result import Result

from app.shared.result import ErrorKind, FeatureError, http_status_for_kind


class CrawlerCode(StrEnum):
    INVALID_SEARCH = "CRAWLER_INVALID_SEARCH"
    SEARCH_UNAVAILABLE = "CRAWLER_SEARCH_UNAVAILABLE"


class CrawlerValidationError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.VALIDATION
    code: ClassVar[CrawlerCode] = CrawlerCode.INVALID_SEARCH


class CrawlerSearchError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.EXTERNAL_SERVICE
    code: ClassVar[CrawlerCode] = CrawlerCode.SEARCH_UNAVAILABLE
    retryable: ClassVar[bool] = True


type CrawlerError = CrawlerValidationError | CrawlerSearchError
type CrawlerResult[T] = Result[T, CrawlerError]


def crawler_error_to_http_status(error: CrawlerError) -> int:
    match error:
        case CrawlerValidationError() | CrawlerSearchError():
            return http_status_for_kind(error.kind, retryable=error.retryable)
        case _ as unreachable:
            assert_never(unreachable)
