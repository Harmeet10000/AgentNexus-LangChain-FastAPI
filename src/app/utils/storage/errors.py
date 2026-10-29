"""Typed failures for the object-storage adapter."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from app.kernel.errors import ErrorKind, FeatureError

if TYPE_CHECKING:
    from typing import ClassVar

    from returns.result import Result


class StorageCode(StrEnum):
    """Error codes owned by the object-storage adapter."""

    INVALID_INPUT = "STORAGE_INVALID_INPUT"
    UNAVAILABLE = "STORAGE_UNAVAILABLE"


class StorageValidationError(FeatureError):
    """The caller supplied an object reference or plan this adapter rejects."""

    kind: ClassVar[ErrorKind] = ErrorKind.VALIDATION
    code: ClassVar[StorageCode] = StorageCode.INVALID_INPUT


class StorageUnavailableError(FeatureError):
    """The bucket or the network failed while serving a storage operation."""

    kind: ClassVar[ErrorKind] = ErrorKind.INFRASTRUCTURE
    code: ClassVar[StorageCode] = StorageCode.UNAVAILABLE
    retryable: ClassVar[bool] = True
    operation: str


type StorageError = StorageValidationError | StorageUnavailableError
type StorageResult[T] = Result[T, StorageError]
