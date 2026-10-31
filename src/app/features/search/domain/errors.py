"""Site-search feature typed errors (ADR-001 closed union)."""

from enum import StrEnum
from typing import ClassVar, assert_never

from returns.result import Result

from app.kernel.errors import ErrorKind, FeatureError
from app.kernel.result import http_status_for_kind


class SearchCode(StrEnum):
    INVALID_REQUEST = "SEARCH_INVALID_REQUEST"
    DOCUMENT_NOT_FOUND = "SEARCH_DOCUMENT_NOT_FOUND"
    TASK_NOT_FOUND = "SEARCH_TASK_NOT_FOUND"
    DOCUMENT_CONFLICT = "SEARCH_DOCUMENT_CONFLICT"
    INFRASTRUCTURE_ERROR = "SEARCH_INFRASTRUCTURE_ERROR"


class SearchValidationError(FeatureError):
    """The request or an identifier is malformed."""

    kind: ClassVar[ErrorKind] = ErrorKind.VALIDATION
    code: ClassVar[SearchCode] = SearchCode.INVALID_REQUEST
    retryable: ClassVar[bool] = False


class SearchNotFoundError(FeatureError):
    """No tenant-visible document or ingest task matches the request."""

    kind: ClassVar[ErrorKind] = ErrorKind.NOT_FOUND
    code: ClassVar[SearchCode] = SearchCode.DOCUMENT_NOT_FOUND
    retryable: ClassVar[bool] = False


class SearchConflictError(FeatureError):
    """A uniqueness constraint (content hash) rejected the write."""

    kind: ClassVar[ErrorKind] = ErrorKind.CONFLICT
    code: ClassVar[SearchCode] = SearchCode.DOCUMENT_CONFLICT
    retryable: ClassVar[bool] = False


class SearchInfrastructureError(FeatureError):
    """Database, cache, queue, or embedding-provider failure.

    Retryable: these are outage-class conditions, so the transport renders 503.
    """

    kind: ClassVar[ErrorKind] = ErrorKind.INFRASTRUCTURE
    code: ClassVar[SearchCode] = SearchCode.INFRASTRUCTURE_ERROR
    retryable: ClassVar[bool] = True


type SearchError = (
    SearchValidationError | SearchNotFoundError | SearchConflictError | SearchInfrastructureError
)
type SearchResult[T] = Result[T, SearchError]


def search_error_to_http_status(error: SearchError) -> int:
    """Exhaustive owner-local classifier for the search error union."""
    match error:
        case SearchValidationError() | SearchNotFoundError() | SearchConflictError():
            return http_status_for_kind(error.kind)
        case SearchInfrastructureError():
            return http_status_for_kind(error.kind, retryable=error.retryable)
        case _ as unreachable:
            assert_never(unreachable)
