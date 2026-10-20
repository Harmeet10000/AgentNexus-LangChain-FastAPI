"""Typed failures shared by billing feature orchestration seams."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from app.kernel.errors import ErrorKind, FeatureError

from .providers import (  # noqa: TC001 -- Pydantic resolves typed failure fields at runtime.
    ProviderErrorCode,
    ProviderOperationKind,
)

if TYPE_CHECKING:
    from typing import ClassVar


class CoordinatorCode(StrEnum):
    PERSISTENCE_ERROR = "LIFECYCLE_PERSISTENCE_ERROR"
    PROVIDER_ERROR = "LIFECYCLE_PROVIDER_ERROR"
    VALIDATION_ERROR = "LIFECYCLE_VALIDATION_ERROR"
    VERSION_CONFLICT = "LIFECYCLE_VERSION_CONFLICT"


class LifecycleCoordinatorPersistenceError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.INFRASTRUCTURE
    code: ClassVar[CoordinatorCode] = CoordinatorCode.PERSISTENCE_ERROR
    retryable: ClassVar[bool] = True

    operation: str | None = None


class LifecycleCoordinatorProviderError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.EXTERNAL_SERVICE
    code: ClassVar[CoordinatorCode] = CoordinatorCode.PROVIDER_ERROR
    retryable: ClassVar[bool] = False

    provider_code: ProviderErrorCode | None = None
    operation: ProviderOperationKind | None = None
    correlation_id: str | None = None
    requires_reconciliation: bool = False


class LifecycleCoordinatorTransientProviderError(LifecycleCoordinatorProviderError):
    """A provider failure that is safe to retry after its backoff."""

    retryable: ClassVar[bool] = True


class LifecycleCoordinatorValidationError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.VALIDATION
    code: ClassVar[CoordinatorCode] = CoordinatorCode.VALIDATION_ERROR
    retryable: ClassVar[bool] = False


class LifecycleCoordinatorVersionConflictError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.CONFLICT
    code: ClassVar[CoordinatorCode] = CoordinatorCode.VERSION_CONFLICT
    retryable: ClassVar[bool] = False

    subscription_id: str | None = None
    expected_version: int | None = None


type LifecycleCoordinatorError = (
    LifecycleCoordinatorPersistenceError
    | LifecycleCoordinatorProviderError
    | LifecycleCoordinatorTransientProviderError
    | LifecycleCoordinatorValidationError
    | LifecycleCoordinatorVersionConflictError
)
