"""
LangSmith observability bootstrap and custom callbacks.
Must be imported before any LangChain/LangGraph objects are created.
"""

from __future__ import annotations

import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler
from langsmith import Client

from app.config import get_settings


def configure_langsmith() -> Client | None:
    """
    Bootstrap LangSmith tracing by setting env vars.
    Call this at application startup, before any agents are built.
    """
    settings = get_settings()
    return Client(
        api_url=settings.LANGSMITH_ENDPOINT, api_key=settings.LANGSMITH_API_KEY
    )


class LatencyCallbackHandler(BaseCallbackHandler):
    """Tracks per-run latency for structured logging."""

    def __init__(self) -> None:
        self._start: dict[UUID, float] = {}

    def on_llm_start(self, *args: Any, run_id: UUID, **kwargs: Any) -> None:
        self._start[run_id] = time.perf_counter()

    def on_llm_end(self, *args: Any, run_id: UUID, **kwargs: Any) -> None:
        elapsed = time.perf_counter() - self._start.pop(run_id, time.perf_counter())
        import logging
        logging.getLogger(__name__).info("llm_latency_ms=%.1f", elapsed * 1000)

    def on_llm_error(self, *args: Any, run_id: UUID, **kwargs: Any) -> None:
        self._start.pop(run_id, None)


class TokenUsageCallbackHandler(BaseCallbackHandler):
    """Accumulates token usage across an entire request for billing/monitoring."""

    def __init__(self) -> None:
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        for gen in response.generations:
            for g in gen:
                usage = getattr(g.message, "usage_metadata", None)
                if usage:
                    self.prompt_tokens += usage.get("input_tokens", 0)
                    self.completion_tokens += usage.get("output_tokens", 0)


class AsyncStreamingCallbackHandler(AsyncCallbackHandler):
    """
    Async callback that forwards tokens to an asyncio.Queue.
    Use in FastAPI SSE endpoints.

    Usage:
        handler = AsyncStreamingCallbackHandler()
        # pass handler to agent invoke config
        async for token in handler:
            yield token
    """

    def __init__(self) -> None:
        import asyncio
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        await self._queue.put(token)

    async def on_llm_end(self, *args: Any, **kwargs: Any) -> None:
        await self._queue.put(None)  # sentinel

    async def on_llm_error(self, *args: Any, **kwargs: Any) -> None:
        await self._queue.put(None)

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        token = await self._queue.get()
        if token is None:
            raise StopAsyncIteration
        return token
