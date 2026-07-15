"""Typed pattern matching helpers for returns.Result.

ty doesn't narrow types through `match`/`case` on returns.Result.Success/Failure.
These helpers provide explicit narrowing for type checker compatibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from returns.result import Failure, Success

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from returns.result import Result

    from .errors import AppError

T = TypeVar("T")
E = TypeVar("E")


def unwrap_success[T, E](result: Result[T, E]) -> T:
    """Unwrap a Success, asserting it's not a Failure.

    Use after `match result: case Success(_):` when ty doesn't narrow.
    """
    if isinstance(result, Success):
        return result.unwrap()  # type: ignore
    msg = "Expected Success but got Failure"
    raise RuntimeError(msg)


def unwrap_failure[T, E](result: Result[T, E]) -> E:
    """Unwrap a Failure, asserting it's not a Success.

    Use after `match result: case Failure(_):` when ty doesn't narrow.
    """
    if isinstance(result, Failure):
        return result.failure()  # type: ignore
    msg = "Expected Failure but got Success"
    raise RuntimeError(msg)


def try_unwrap_success[T, E](result: Result[T, E]) -> T | None:
    """Return the success value or None if Failure.

    Useful when you need the value but don't want to raise.
    """
    if isinstance(result, Success):
        return result.unwrap()  # type: ignore
    return None


def try_unwrap_failure[T, E](result: Result[T, E]) -> E | None:
    """Return the failure error or None if Success.

    Useful when you need the error but don't want to raise.
    """
    if isinstance(result, Failure):
        return result.failure()  # type: ignore
    return None


def map_success[T, E](result: Result[T, E], func: Callable[[T], Any]) -> Result[Any, E]:
    """Map a function over the success value, passing through failures.

    Type-safe alternative to `result.map(func)` when ty needs help.
    """
    if isinstance(result, Success):
        return Success(func(result.unwrap()))  # type: ignore
    elif isinstance(result, Failure):
        return result
    raise RuntimeError(f"Unexpected Result variant: {type(result)}")


def bind_success[T, E](result: Result[T, E], func: Callable[[T], Result[Any, E]]) -> Result[Any, E]:
    """Bind a Result-returning function over the success value.

    Type-safe alternative to `result.bind(func)` when ty needs help.
    """
    if isinstance(result, Success):
        return func(result.unwrap())  # type: ignore
    elif isinstance(result, Failure):
        return result
    raise RuntimeError(f"Unexpected Result variant: {type(result)}")


def match_result[T, E, U](
    result: Result[T, E],
    on_success: Callable[[T], U],
    on_failure: Callable[[E], U],
) -> U:
    """Match a Result with two handlers, returning a unified type.

    This is the recommended pattern for exhaustive matching with proper
    type narrowing for both branches.
    """
    if isinstance(result, Success):
        return on_success(result.unwrap())  # type: ignore
    elif isinstance(result, Failure):
        return on_failure(result.failure())  # type: ignore
    raise RuntimeError(f"Unexpected Result variant: {type(result)}")


def match_result_or_raise[T, E](
    result: Result[T, E],
    error_factory: Callable[[E], Exception] | None = None,
) -> T:
    """Unwrap success or raise an exception from the failure.

    Args:
        result: The Result to unwrap
        error_factory: Optional function to convert the error to an exception.
                       Defaults to raising the error directly if it's an Exception.
    """
    if isinstance(result, Success):
        return result.unwrap()  # type: ignore
    elif isinstance(result, Failure):
        error = result.failure()
        if error_factory:
            raise error_factory(error)
        if isinstance(error, Exception):
            raise error
        raise RuntimeError(str(error))
    else:
        raise RuntimeError(f"Unexpected Result variant: {type(result)}")


# --- AppError-specific helpers ---


def unwrap_app_success[T](result: Result[T, AppError]) -> T:
    """Unwrap a Result[_, AppError] Success after pattern match."""
    if isinstance(result, Success):
        return result.unwrap()  # type: ignore
    raise RuntimeError("Expected Success but got Failure")


def unwrap_app_failure[T](result: Result[T, AppError]) -> AppError:
    """Unwrap a Result[_, AppError] Failure after pattern match."""
    if isinstance(result, Failure):
        return result.failure()
    raise RuntimeError("Expected Failure but got Success")


def match_app_result[T, U](
    result: Result[T, AppError],
    on_success: Callable[[T], U],
    on_failure: Callable[[AppError], U],
) -> U:
    """Match a Result[_, AppError] with properly typed handlers."""
    return match_result(result, on_success, on_failure)


def unwrap_app_result_or_raise[T](result: Result[T, AppError]) -> T:
    """Unwrap Result[_, AppError] or raise the AppError as an exception."""
    return match_result_or_raise(result, lambda e: e)
