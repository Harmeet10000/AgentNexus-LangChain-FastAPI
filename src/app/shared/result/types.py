"""Shared Result type aliases for expected internal failures."""

from returns.future import FutureResult
from returns.result import Result

from .errors import AppError

type AppResult[T] = Result[T, AppError]
type AppFutureResult[T] = FutureResult[T, AppError]
