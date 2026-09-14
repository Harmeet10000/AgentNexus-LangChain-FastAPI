"""Lightweight public lifecycle entry points.

Keeping the implementation imports inside these wrappers lets provider modules
be imported by worker children and tooling without bootstrapping the full API
dependency graph.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Delegate to the application lifespan without importing it eagerly."""
    from .lifespan import lifespan as lifespan_impl  # noqa: PLC0415

    async with lifespan_impl(app):
        yield


def setup_signal_handlers() -> None:
    """Delegate signal setup without loading logging during package import."""
    from .signals import setup_signal_handlers as setup_impl  # noqa: PLC0415

    setup_impl()


__all__ = [
    "lifespan",
    "setup_signal_handlers",
]
