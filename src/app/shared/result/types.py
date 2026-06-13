"""Shared Result type aliases for expected internal failures."""

from returns.result import Result

from .errors import AppError

type AppResult[T] = Result[T, AppError]
