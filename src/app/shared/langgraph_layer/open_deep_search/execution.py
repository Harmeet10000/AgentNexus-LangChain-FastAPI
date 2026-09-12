"""Concurrency controls for immediate deep-research provider calls."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable

    from langchain_core.runnables import RunnableConfig


class ResearchExecutionGate:
    """Limit provider work shared by deep-research runs in one process."""

    def __init__(
        self,
        *,
        max_concurrent_search_requests: int = 10,
        max_concurrent_summaries: int = 10,
    ) -> None:
        if max_concurrent_search_requests < 1:
            msg = "max_concurrent_search_requests must be at least 1"
            raise ValueError(msg)
        if max_concurrent_summaries < 1:
            msg = "max_concurrent_summaries must be at least 1"
            raise ValueError(msg)
        self._search_slots = asyncio.Semaphore(max_concurrent_search_requests)
        self._summary_slots = asyncio.Semaphore(max_concurrent_summaries)

    async def run_search[T](self, operation: Callable[[], Awaitable[T]]) -> T:
        """Run one search operation while holding a process-local search slot."""
        async with self._search_slots:
            return await operation()

    async def run_summary[T](self, operation: Callable[[], Awaitable[T]]) -> T:
        """Run one summarization operation while holding a process-local slot."""
        async with self._summary_slots:
            return await operation()


_DEFAULT_RESEARCH_EXECUTION_GATE = ResearchExecutionGate()


def get_research_execution_gate(config: RunnableConfig | None = None) -> ResearchExecutionGate:
    """Return the injected gate or the process-local default gate."""
    if config:
        configurable = config.get("configurable", {})
        gate = configurable.get("research_execution_gate")
        if isinstance(gate, ResearchExecutionGate):
            return gate
    return _DEFAULT_RESEARCH_EXECUTION_GATE


async def gather_limited[T](
    operations: Iterable[Callable[[], Awaitable[T]]],
    *,
    limit: int,
) -> list[T]:
    """Run awaitable factories with a per-invocation concurrency limit."""
    if limit < 1:
        msg = "limit must be at least 1"
        raise ValueError(msg)

    semaphore = asyncio.Semaphore(limit)

    async def run(operation: Callable[[], Awaitable[T]]) -> T:
        async with semaphore:
            return await operation()

    tasks = [asyncio.create_task(run(operation)) for operation in operations]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
