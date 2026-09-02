"""Diagnostics attached to third-party exceptions before typed conversion."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

from sqlalchemy.exc import IntegrityError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from sqlalchemy.exc import SQLAlchemyError


def add_database_error_note(
    exc: SQLAlchemyError,
    *,
    table: str,
    operation: str | None = None,
    context: Mapping[str, object] | None = None,
) -> None:
    """Attach bounded SQLAlchemy context before the exception is converted.

    The original exception is not retained by a typed ``Failure``. Its note is
    therefore the last diagnostic channel available at the driver boundary.
    Constraint names are read from PostgreSQL's optional ``diag`` object when
    present; synthetic and non-PostgreSQL exceptions receive ``unknown``.
    """
    frame = inspect.currentframe()
    caller = frame.f_back if frame is not None else None
    operation_name = operation or (caller.f_code.co_name if caller is not None else "unknown")
    parts = [f"table={table}", f"operation={operation_name}", f"query={operation_name}"]
    if isinstance(exc, IntegrityError):
        original = getattr(exc, "orig", None)
        diagnostic = getattr(original, "diag", None)
        constraint_name = getattr(diagnostic, "constraint_name", None) or "unknown"
        parts.append(f"constraint_name={constraint_name}")
    if context:
        parts.extend(f"{key}={_bounded(value)}" for key, value in context.items())
    exc.add_note(", ".join(parts))


def _bounded(value: object, *, limit: int = 160) -> str:
    text = str(value)
    return text if len(text) <= limit else f"{text[:limit]}..."
