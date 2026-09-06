"""Typed failures for the Redis cache adapter."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from app.shared.result import ErrorKind, FeatureError

if TYPE_CHECKING:
    from typing import ClassVar

    from returns.result import Result


class CacheCode(StrEnum):
    """Error codes owned by the cache adapter."""

    OPERATION_FAILED = "CACHE_OPERATION_FAILED"


class CacheError(FeatureError):
    """A Redis/cache operation failed at the adapter boundary."""

    kind: ClassVar[ErrorKind] = ErrorKind.INFRASTRUCTURE
    code: ClassVar[CacheCode] = CacheCode.OPERATION_FAILED
    retryable: ClassVar[bool] = False


type CacheResult[T] = Result[T, CacheError]
