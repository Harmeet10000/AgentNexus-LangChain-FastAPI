"""Agent Saul typed errors for HTTP-domain operations.

The WebSocket session remains exception-native because close and security
exceptions are transport control flow.
"""

from enum import StrEnum
from typing import ClassVar, assert_never

from returns.result import Result

from app.shared.result import ErrorKind, FeatureError, http_status_for_kind


class AgentSaulCode(StrEnum):
    INVALID_SESSION = "AGENT_SAUL_INVALID_SESSION"
    GRAPH_FAILED = "AGENT_SAUL_GRAPH_FAILED"


class AgentSaulValidationError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.VALIDATION
    code: ClassVar[AgentSaulCode] = AgentSaulCode.INVALID_SESSION
    retryable: ClassVar[bool] = False


class AgentSaulInfrastructureError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.INFRASTRUCTURE
    code: ClassVar[AgentSaulCode] = AgentSaulCode.GRAPH_FAILED
    retryable: ClassVar[bool] = True


type AgentSaulError = AgentSaulValidationError | AgentSaulInfrastructureError
type AgentSaulResult[T] = Result[T, AgentSaulError]


def agent_saul_error_to_http_status(error: AgentSaulError) -> int:
    match error:
        case AgentSaulValidationError() | AgentSaulInfrastructureError():
            return http_status_for_kind(error.kind, retryable=error.retryable)
        case _ as unreachable:
            assert_never(unreachable)
