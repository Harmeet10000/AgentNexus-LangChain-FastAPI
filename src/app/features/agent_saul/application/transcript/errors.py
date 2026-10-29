"""Chat transcript typed errors used as Failure payloads (doc §10).

Mirrors the billing plans idiom exactly (verify item 6):
`TranscriptResult[T] = Result[T, TranscriptError]` is aliased HERE, never in
repository.py — importing the alias from the repository invites circular
imports and hand-inlined unions.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, assert_never

from returns.result import Result

from app.kernel.errors import ErrorKind, FeatureError
from app.kernel.result import http_status_for_kind

if TYPE_CHECKING:
    from typing import ClassVar


class TranscriptCode(StrEnum):
    NOT_FOUND = "TRANSCRIPT_NOT_FOUND"
    SCHEMA_VERSION = "TRANSCRIPT_SCHEMA_VERSION"
    SESSION_NOT_FOUND = "TRANSCRIPT_SESSION_NOT_FOUND"
    DATABASE_ERROR = "TRANSCRIPT_DATABASE_ERROR"


class TranscriptNotFoundError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.NOT_FOUND
    code: ClassVar[TranscriptCode] = TranscriptCode.NOT_FOUND
    retryable: ClassVar[bool] = False

    session_id: str


class TranscriptSchemaError(FeatureError):
    """Unknown transcript schema_version at a read boundary — fail closed (§9)."""

    kind: ClassVar[ErrorKind] = ErrorKind.VALIDATION
    code: ClassVar[TranscriptCode] = TranscriptCode.SCHEMA_VERSION
    retryable: ClassVar[bool] = False

    operation: str
    expected_version: int
    found_version: int | None


class TranscriptInfrastructureError(FeatureError):
    """Driver errors mapped to this at the repository edge; sanitized payload only."""

    kind: ClassVar[ErrorKind] = ErrorKind.INFRASTRUCTURE
    code: ClassVar[TranscriptCode] = TranscriptCode.DATABASE_ERROR
    retryable: ClassVar[bool] = True

    operation: str


type TranscriptError = (
    TranscriptNotFoundError | TranscriptSchemaError | TranscriptInfrastructureError
)
type TranscriptResult[T] = Result[T, TranscriptError]


def transcript_error_to_http_status(error: TranscriptError) -> int:
    match error:
        case TranscriptNotFoundError() | TranscriptSchemaError():
            return http_status_for_kind(error.kind, retryable=error.retryable)
        case TranscriptInfrastructureError():
            return http_status_for_kind(error.kind, retryable=error.retryable)
        case _ as unreachable:
            assert_never(unreachable)
