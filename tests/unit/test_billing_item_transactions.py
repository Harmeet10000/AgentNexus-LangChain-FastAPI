"""Regression tests for independent billing batch-item transactions."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from app.connections.postgres import independent_session


def _run(coro):
    return asyncio.run(coro)


class _SessionContext:
    def __init__(self, session: AsyncMock) -> None:
        self.session = session

    async def __aenter__(self) -> AsyncMock:
        return self.session

    async def __aexit__(self, *_args: object) -> None:
        return None


def test_later_item_rollback_does_not_discard_prior_commit() -> None:
    first = AsyncMock()
    second = AsyncMock()
    sessions = iter((first, second))
    session_factory = MagicMock(side_effect=lambda: _SessionContext(next(sessions)))

    async def run_batch() -> None:
        async with independent_session(session_factory):
            pass

        async with independent_session(session_factory) as item_session:
            # Repository Failure paths roll back their own session before returning.
            await item_session.rollback()

    _run(run_batch())

    assert session_factory.call_count == 2
    first.commit.assert_awaited_once()
    first.rollback.assert_not_awaited()
    second.rollback.assert_awaited_once()
    second.commit.assert_awaited_once()
