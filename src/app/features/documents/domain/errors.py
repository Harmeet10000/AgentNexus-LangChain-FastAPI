"""Documents feature typed errors."""

from enum import StrEnum
from typing import ClassVar, assert_never

from returns.result import Result

from app.kernel.errors import ErrorKind, FeatureError
from app.shared.result.render import http_status_for_kind


class DocumentCode(StrEnum):
    DOCUMENT_NOT_FOUND = "DOCUMENT_NOT_FOUND"
    STATUS_NOT_FOUND = "STATUS_NOT_FOUND"
    DOCUMENT_CONFLICT = "DOCUMENT_CONFLICT"
    CHUNK_CONFLICT = "CHUNK_CONFLICT"
    INVALID_DOCUMENT = "INVALID_DOCUMENT"
    STORAGE_UNAVAILABLE = "STORAGE_UNAVAILABLE"
    DATABASE_ERROR = "DATABASE_ERROR"
    EMBEDDING_WIDTH_MISMATCH = "EMBEDDING_WIDTH_MISMATCH"
    GRAPH_WRITE_FAILED = "GRAPH_WRITE_FAILED"
    INGESTION_CHECKPOINT_FAILED = "INGESTION_CHECKPOINT_FAILED"
    # The D1 policy-gate refusal code (harden 7.1/7.2): the exact typed failure
    # every delete attempt returns while the policy is unapproved.
    DELETION_POLICY_NOT_APPROVED = "deletion_policy_not_approved"


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


class DocumentGraphWriteError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.INFRASTRUCTURE
    code: ClassVar[DocumentCode] = DocumentCode.GRAPH_WRITE_FAILED
    retryable: ClassVar[bool] = True


class DocumentIngestionCheckpointError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.INFRASTRUCTURE
    code: ClassVar[DocumentCode] = DocumentCode.INGESTION_CHECKPOINT_FAILED
    retryable: ClassVar[bool] = True


class DeletionPolicyNotApprovedError(FeatureError):
    """The D1 policy gate is closed: deletion is disabled and nothing was removed.

    The D1 draft (`notes/deletion-policy-draft.md`, Decision 5) and the
    document-delete-and-access-audit delta require a typed refusal — never a
    silent no-op or a 500 — while the retention/recovery policy and the L1
    legal/security sign-off are unapproved.
    """

    kind: ClassVar[ErrorKind] = ErrorKind.AUTHORIZATION
    code: ClassVar[DocumentCode] = DocumentCode.DELETION_POLICY_NOT_APPROVED
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
    | DocumentGraphWriteError
    | DocumentIngestionCheckpointError
    | DeletionPolicyNotApprovedError
)
type DocumentResult[T] = Result[T, DocumentError]


def document_error_to_http_status(error: DocumentError) -> int:
    match error:
        case DocumentNotFoundError() | DocumentStatusNotFoundError():
            return http_status_for_kind(error.kind)
        case DocumentConflictError() | DocumentChunkConflictError():
            return http_status_for_kind(error.kind)
        case DocumentValidationError() | DeletionPolicyNotApprovedError():
            return http_status_for_kind(error.kind)
        case (
            DocumentStorageError()
            | DocumentDatabaseError()
            | DocumentEmbeddingWidthError()
            | DocumentGraphWriteError()
            | DocumentIngestionCheckpointError()
        ):
            return http_status_for_kind(error.kind, retryable=error.retryable)
        case _ as unreachable:
            assert_never(unreachable)
