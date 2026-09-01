"""Documents feature typed errors."""

from enum import StrEnum
from typing import ClassVar, assert_never

from returns.result import Result

from app.shared.result import ErrorKind, FeatureError, http_status_for_kind


class DocumentCode(StrEnum):
    DOCUMENT_NOT_FOUND = "DOCUMENT_NOT_FOUND"
    STATUS_NOT_FOUND = "STATUS_NOT_FOUND"
    DOCUMENT_CONFLICT = "DOCUMENT_CONFLICT"
    CHUNK_CONFLICT = "CHUNK_CONFLICT"
    INVALID_DOCUMENT = "INVALID_DOCUMENT"
    STORAGE_UNAVAILABLE = "STORAGE_UNAVAILABLE"
    DATABASE_ERROR = "DATABASE_ERROR"
    EMBEDDING_WIDTH_MISMATCH = "EMBEDDING_WIDTH_MISMATCH"


class DocumentNotFoundError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.NOT_FOUND
    code: ClassVar[DocumentCode] = DocumentCode.DOCUMENT_NOT_FOUND
    retryable: ClassVar[bool] = False


class DocumentStatusNotFoundError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.NOT_FOUND
    code: ClassVar[DocumentCode] = DocumentCode.STATUS_NOT_FOUND
    retryable: ClassVar[bool] = False


class DocumentConflictError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.CONFLICT
    code: ClassVar[DocumentCode] = DocumentCode.DOCUMENT_CONFLICT
    retryable: ClassVar[bool] = False


class DocumentChunkConflictError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.CONFLICT
    code: ClassVar[DocumentCode] = DocumentCode.CHUNK_CONFLICT
    retryable: ClassVar[bool] = False


class DocumentValidationError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.VALIDATION
    code: ClassVar[DocumentCode] = DocumentCode.INVALID_DOCUMENT
    retryable: ClassVar[bool] = False


class DocumentStorageError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.INFRASTRUCTURE
    code: ClassVar[DocumentCode] = DocumentCode.STORAGE_UNAVAILABLE
    retryable: ClassVar[bool] = True


class DocumentDatabaseError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.INFRASTRUCTURE
    code: ClassVar[DocumentCode] = DocumentCode.DATABASE_ERROR
    retryable: ClassVar[bool] = False


class DocumentEmbeddingWidthError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.INFRASTRUCTURE
    code: ClassVar[DocumentCode] = DocumentCode.EMBEDDING_WIDTH_MISMATCH
    retryable: ClassVar[bool] = False


type DocumentError = (
    DocumentNotFoundError
    | DocumentStatusNotFoundError
    | DocumentConflictError
    | DocumentChunkConflictError
    | DocumentValidationError
    | DocumentStorageError
    | DocumentDatabaseError
    | DocumentEmbeddingWidthError
)
type DocumentResult[T] = Result[T, DocumentError]


def document_error_to_http_status(error: DocumentError) -> int:
    match error:
        case DocumentNotFoundError() | DocumentStatusNotFoundError():
            return http_status_for_kind(error.kind)
        case DocumentConflictError() | DocumentChunkConflictError():
            return http_status_for_kind(error.kind)
        case DocumentValidationError():
            return http_status_for_kind(error.kind)
        case DocumentStorageError() | DocumentDatabaseError() | DocumentEmbeddingWidthError():
            return http_status_for_kind(error.kind, retryable=error.retryable)
        case _ as unreachable:
            assert_never(unreachable)
