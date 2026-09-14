"""Typed failures owned by the shared retrieval-evaluation boundary."""

from __future__ import annotations

from enum import StrEnum
from typing import ClassVar  # noqa: TC003 -- Pydantic resolves ClassVar at runtime

from app.shared.result import ErrorKind, FeatureError


class EvaluationCode(StrEnum):
    MALFORMED_GOLDEN_ROW = "MALFORMED_GOLDEN_ROW"


class MalformedGoldenRowError(FeatureError):
    """Expected failure for invalid JSON or schema in one golden-set row."""

    kind: ClassVar[ErrorKind] = ErrorKind.VALIDATION
    code: ClassVar[EvaluationCode] = EvaluationCode.MALFORMED_GOLDEN_ROW
    retryable: ClassVar[bool] = False

    row_index: int
