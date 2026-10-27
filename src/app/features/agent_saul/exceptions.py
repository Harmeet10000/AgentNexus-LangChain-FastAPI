"""Agent Saul typed errors for HTTP-domain operations.

The WebSocket session remains exception-native because close and security
exceptions are transport control flow.
"""

from enum import StrEnum
from typing import ClassVar, assert_never

from returns.result import Result

from app.kernel.errors import ErrorKind, FeatureError
from app.kernel.result import http_status_for_kind


class AgentSaulCode(StrEnum):
    INVALID_SESSION = "AGENT_SAUL_INVALID_SESSION"
    GRAPH_FAILED = "AGENT_SAUL_GRAPH_FAILED"
    DOCUMENT_NOT_FOUND = "AGENT_SAUL_DOCUMENT_NOT_FOUND"
    DOCUMENT_NOT_READY = "AGENT_SAUL_DOCUMENT_NOT_READY"
    DOCUMENT_INGESTION_FAILED = "AGENT_SAUL_DOCUMENT_INGESTION_FAILED"


class AgentSaulValidationError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.VALIDATION
    code: ClassVar[AgentSaulCode] = AgentSaulCode.INVALID_SESSION
    retryable: ClassVar[bool] = False


class AgentSaulInfrastructureError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.INFRASTRUCTURE
    code: ClassVar[AgentSaulCode] = AgentSaulCode.GRAPH_FAILED
    retryable: ClassVar[bool] = True


class AgentSaulDocumentNotFoundError(FeatureError):
    """Session pre-flight rejected a doc_id the user has no row for.

    The repository read is user-scoped, so a nonexistent and a foreign
    doc_id are the same refusal — reveal nothing, answering only 404.
    """

    kind: ClassVar[ErrorKind] = ErrorKind.NOT_FOUND
    code: ClassVar[AgentSaulCode] = AgentSaulCode.DOCUMENT_NOT_FOUND
    retryable: ClassVar[bool] = False

    doc_id: str
    user_id: str


class AgentSaulDocumentNotReadyError(FeatureError):
    """Selected document exists but has not reached an admissible state."""

    kind: ClassVar[ErrorKind] = ErrorKind.CONFLICT
    code: ClassVar[AgentSaulCode] = AgentSaulCode.DOCUMENT_NOT_READY
    retryable: ClassVar[bool] = True

    doc_id: str
    status: str


class AgentSaulDocumentIngestionFailedError(FeatureError):
    """Selected document has a terminal ingestion failure."""

    kind: ClassVar[ErrorKind] = ErrorKind.VALIDATION
    code: ClassVar[AgentSaulCode] = AgentSaulCode.DOCUMENT_INGESTION_FAILED
    retryable: ClassVar[bool] = False

    doc_id: str
    status: str


type AgentSaulError = (
    AgentSaulValidationError
    | AgentSaulInfrastructureError
    | AgentSaulDocumentNotFoundError
    | AgentSaulDocumentNotReadyError
    | AgentSaulDocumentIngestionFailedError
)
type AgentSaulResult[T] = Result[T, AgentSaulError]


def agent_saul_error_to_http_status(error: AgentSaulError) -> int:
    match error:
        case (
            AgentSaulValidationError()
            | AgentSaulInfrastructureError()
            | AgentSaulDocumentNotFoundError()
            | AgentSaulDocumentNotReadyError()
            | AgentSaulDocumentIngestionFailedError()
        ):
            return http_status_for_kind(error.kind, retryable=error.retryable)
        case _ as unreachable:
            assert_never(unreachable)
