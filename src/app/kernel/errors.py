"""Pure error spine: the only cross-feature failure vocabulary.

No I/O, no FastAPI, no SQLAlchemy. Feature slices subclass `FeatureError`
directly; HTTP status mapping lives in `app.shared.result` (render path).
"""

from enum import StrEnum
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

type ErrorDetails = dict[str, object]


class ErrorKind(StrEnum):
    """Shared classification vocabulary — the only cross-feature error vocabulary."""

    VALIDATION = "validation"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    INFRASTRUCTURE = "infrastructure"
    EXTERNAL_SERVICE = "external_service"


class FeatureError(BaseModel):
    """Base for per-feature typed errors — classification is a ClassVar, never a field."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    message: str
    details: ErrorDetails | None = None
    source: str | None = None

    kind: ClassVar[ErrorKind]
    code: ClassVar[StrEnum]
    retryable: ClassVar[bool] = False
