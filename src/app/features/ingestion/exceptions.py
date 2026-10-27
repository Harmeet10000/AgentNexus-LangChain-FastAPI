"""Ingestion feature typed errors."""

from enum import StrEnum
from typing import ClassVar, assert_never

from returns.result import Result

from app.kernel.errors import ErrorKind, FeatureError
from app.kernel.result import http_status_for_kind


class IngestionCode(StrEnum):
    INVALID_DOCUMENT = "INVALID_DOCUMENT"
    GRAPH_FAILED = "INGESTION_GRAPH_FAILED"
    PIPELINE_FAILED = "INGESTION_PIPELINE_FAILED"
    INTERNAL_ERROR = "INGESTION_INTERNAL_ERROR"


class IngestionGraphError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.INFRASTRUCTURE
    code: ClassVar[IngestionCode] = IngestionCode.GRAPH_FAILED
    retryable: ClassVar[bool] = True

    doc_id: str


class IngestionPipelineError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.INFRASTRUCTURE
    code: ClassVar[IngestionCode] = IngestionCode.PIPELINE_FAILED
    retryable: ClassVar[bool] = False

    doc_id: str


class IngestionInternalError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.INFRASTRUCTURE
    code: ClassVar[IngestionCode] = IngestionCode.INTERNAL_ERROR
    retryable: ClassVar[bool] = False

    doc_id: str


class IngestionValidationError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.VALIDATION
    code: ClassVar[IngestionCode] = IngestionCode.INVALID_DOCUMENT
    retryable: ClassVar[bool] = False

    doc_id: str = ""


type IngestionError = (
    IngestionGraphError | IngestionPipelineError | IngestionInternalError | IngestionValidationError
)
type IngestionResult[T] = Result[T, IngestionError]


def ingestion_error_to_http_status(error: IngestionError) -> int:
    match error:
        case IngestionGraphError() | IngestionPipelineError() | IngestionInternalError():
            return http_status_for_kind(error.kind, retryable=error.retryable)
        case IngestionValidationError():
            return http_status_for_kind(error.kind, retryable=error.retryable)
        case _ as unreachable:
            assert_never(unreachable)
