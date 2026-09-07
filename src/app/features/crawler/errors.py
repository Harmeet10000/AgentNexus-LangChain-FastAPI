"""Crawler feature typed errors."""

from enum import StrEnum
from typing import ClassVar, assert_never

from returns.result import Result

from app.shared.result import ErrorKind, FeatureError, http_status_for_kind


class CrawlerCode(StrEnum):
    INVALID_SEARCH = "CRAWLER_INVALID_SEARCH"
    SEARCH_UNAVAILABLE = "CRAWLER_SEARCH_UNAVAILABLE"
    CRAWL_UNAVAILABLE = "CRAWLER_CRAWL_UNAVAILABLE"


class CrawlerValidationError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.VALIDATION
    code: ClassVar[CrawlerCode] = CrawlerCode.INVALID_SEARCH


class CrawlerSearchError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.EXTERNAL_SERVICE
    code: ClassVar[CrawlerCode] = CrawlerCode.SEARCH_UNAVAILABLE
    retryable: ClassVar[bool] = True


class CrawlerCrawlError(FeatureError):
    """The crawl provider could not complete the requested crawl."""

    kind: ClassVar[ErrorKind] = ErrorKind.EXTERNAL_SERVICE
    code: ClassVar[CrawlerCode] = CrawlerCode.CRAWL_UNAVAILABLE
    retryable: ClassVar[bool] = True


type CrawlerError = CrawlerValidationError | CrawlerSearchError | CrawlerCrawlError
type CrawlerResult[T] = Result[T, CrawlerError]


def crawler_error_to_http_status(error: CrawlerError) -> int:
    match error:
        case CrawlerValidationError() | CrawlerSearchError() | CrawlerCrawlError():
            return http_status_for_kind(error.kind, retryable=error.retryable)
        case _ as unreachable:
            assert_never(unreachable)
