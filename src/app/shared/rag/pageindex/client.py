"""Async helpers for PageIndex with concurrency and backpressure controls."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Protocol

from app.utils.logger import logger

if TYPE_CHECKING:
    from collections.abc import Sequence


class PageIndexClientProtocol(Protocol):
    """Protocol for the small subset of client methods used in this module."""

    def chat_completions(self, **kwargs: object) -> object: ...


@dataclass(slots=True, frozen=True)
class PageIndexConfig:
    """Configuration for indexing operations."""

    api_key: str | None = None
    model: str = "gpt-4o-2024-11-20"
    toc_check_page_num: int = 20
    max_page_num_each_node: int = 10
    max_token_num_each_node: int = 20_000
    if_add_node_id: str = "yes"
    if_add_node_summary: str = "yes"
    if_add_doc_description: str = "no"
    if_add_node_text: str = "no"
    additional_kwargs: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class PageIndexBatchConfig:
    """Concurrency settings for indexing batches."""

    max_concurrency: int = 4
    return_exceptions: bool = True


@dataclass(slots=True, frozen=True)
class PageIndexChatConfig:
    """Configuration for PageIndex chat completion calls."""

    api_key: str | None = None
    model: str | None = None
    stream: bool = False
    temperature: float | None = None
    additional_kwargs: dict[str, object] = field(default_factory=dict)


@lru_cache(maxsize=1)
def _executor() -> ThreadPoolExecutor:
    default_workers = max((os.cpu_count() or 4) * 2, 8)
    return ThreadPoolExecutor(max_workers=default_workers, thread_name_prefix="pi")


@lru_cache(maxsize=1)
def _load_pageindex_module() -> object:
    try:
        return import_module("pageindex")
    except ImportError as exc:
        raise RuntimeError(
            "pageindex is not installed. Add it to runtime dependencies first."
        ) from exc


@lru_cache(maxsize=32)
def _build_client(api_key: str) -> PageIndexClientProtocol:
    module = _load_pageindex_module()
    if not isinstance(module, ModuleType):
        raise TypeError("Invalid pageindex module instance.")
    return module.PageIndexClient(api_key=api_key)


def _resolve_api_key(api_key: str | None) -> str:
    resolved = api_key or os.getenv("PAGEINDEX_API_KEY", "")
    if not resolved:
        raise ValueError("PAGEINDEX_API_KEY is required for PageIndex operations.")
    return resolved


def _raise_non_iterable_stream_response() -> None:
    raise TypeError("Expected iterable response from streamed chat_completions.")


async def apage_index(
    *,
    doc: str,
    config: PageIndexConfig | None = None,
) -> dict[str, object]:
    """Run PageIndex document indexing without blocking the event loop."""
    runtime = config or PageIndexConfig()
    module = _load_pageindex_module()
    if not isinstance(module, ModuleType):
        raise TypeError("Invalid pageindex module instance.")
    key = _resolve_api_key(runtime.api_key)

    kwargs: dict[str, object] = {
        "doc": doc,
        "model": runtime.model,
        "toc_check_page_num": runtime.toc_check_page_num,
        "max_page_num_each_node": runtime.max_page_num_each_node,
        "max_token_num_each_node": runtime.max_token_num_each_node,
        "if_add_node_id": runtime.if_add_node_id,
        "if_add_node_summary": runtime.if_add_node_summary,
        "if_add_doc_description": runtime.if_add_doc_description,
        "if_add_node_text": runtime.if_add_node_text,
        "api_key": key,
    }
    if runtime.additional_kwargs:
        kwargs.update(runtime.additional_kwargs)

    loop = asyncio.get_running_loop()
    logger.info("Starting PageIndex indexing", doc=doc, model=runtime.model)
    return await loop.run_in_executor(_executor(), lambda: module.page_index(**kwargs))


async def abatch_page_index(
    *,
    docs: Sequence[str],
    config: PageIndexConfig | None = None,
    batch_config: PageIndexBatchConfig | None = None,
) -> list[object]:
    """Index multiple documents concurrently with a bounded in-flight window."""
    if not docs:
        return []

    runtime_config = config or PageIndexConfig()
    runtime_batch = batch_config or PageIndexBatchConfig()
    semaphore = asyncio.Semaphore(max(1, runtime_batch.max_concurrency))

    async def _worker(doc_path: str) -> dict[str, object]:
        async with semaphore:
            return await apage_index(doc=doc_path, config=runtime_config)

    tasks = [asyncio.create_task(_worker(doc_path)) for doc_path in docs]
    return await asyncio.gather(
        *tasks,
        return_exceptions=runtime_batch.return_exceptions,
    )


async def achat_completion(
    *,
    doc_id: str,
    messages: list[dict[str, str]],
    config: PageIndexChatConfig | None = None,
) -> object:
    """Call PageIndex chat completion API in non-streaming mode."""
    runtime = config or PageIndexChatConfig()
    key = _resolve_api_key(runtime.api_key)
    client = _build_client(key)

    kwargs: dict[str, object] = {
        "messages": messages,
        "doc_id": doc_id,
        "stream": False,
    }
    if runtime.model is not None:
        kwargs["model"] = runtime.model
    if runtime.temperature is not None:
        kwargs["temperature"] = runtime.temperature
    if runtime.additional_kwargs:
        kwargs.update(runtime.additional_kwargs)

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _executor(),
        lambda: client.chat_completions(**kwargs),
    )


async def astream_chat_completions(
    *,
    doc_id: str,
    messages: list[dict[str, str]],
    config: PageIndexChatConfig | None = None,
) -> object:
    """
    Stream chat completion chunks from PageIndex as an async generator.

    This bridges the SDK's sync iterator to async code using a background thread and
    an asyncio queue.
    """
    runtime = config or PageIndexChatConfig(stream=True)
    key = _resolve_api_key(runtime.api_key)
    client = _build_client(key)

    kwargs: dict[str, object] = {
        "messages": messages,
        "doc_id": doc_id,
        "stream": True,
    }
    if runtime.model is not None:
        kwargs["model"] = runtime.model
    if runtime.temperature is not None:
        kwargs["temperature"] = runtime.temperature
    if runtime.additional_kwargs:
        kwargs.update(runtime.additional_kwargs)

    queue: asyncio.Queue[object] = asyncio.Queue(maxsize=256)
    loop = asyncio.get_running_loop()
    end_marker = object()

    def _producer() -> None:
        try:
            response = client.chat_completions(**kwargs)
            if not isinstance(response, Iterable):
                _raise_non_iterable_stream_response()
            for chunk in response:
                loop.call_soon_threadsafe(queue.put_nowait, chunk)
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, end_marker)

    task = loop.run_in_executor(_executor(), _producer)
    try:
        while True:
            item = await queue.get()
            if item is end_marker:
                break
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        await task


def create_node_map(tree: dict[str, object]) -> dict[str, dict[str, object]]:
    """Flatten tree nodes into a map keyed by node_id."""
    mapped: dict[str, dict[str, object]] = {}
    root_nodes = tree.get("structure", [])
    if not isinstance(root_nodes, list):
        return mapped

    stack: list[dict[str, object]] = []
    for node in root_nodes:
        if not isinstance(node, dict):
            continue
        normalized = {key: value for key, value in node.items() if isinstance(key, str)}
        stack.append(normalized)

    while stack:
        node = stack.pop()
        node_id = node.get("node_id")
        if isinstance(node_id, str):
            mapped[node_id] = node

        children = node.get("nodes", [])
        if isinstance(children, list):
            for child in children:
                if not isinstance(child, dict):
                    continue
                normalized_child = {
                    key: value for key, value in child.items() if isinstance(key, str)
                }
                stack.append(normalized_child)

    return mapped


def gather_node_text(
    *,
    tree: dict[str, object],
    node_ids: Sequence[str],
    max_chars: int | None = None,
) -> str:
    """Gather text payload for selected nodes in order of requested node IDs."""
    node_map = create_node_map(tree)
    parts: list[str] = []
    total = 0

    for node_id in node_ids:
        node = node_map.get(node_id)
        if node is None:
            continue

        text = node.get("text")
        if not isinstance(text, str) or not text:
            continue

        if max_chars is None:
            parts.append(text)
            continue

        remaining = max_chars - total
        if remaining <= 0:
            break
        chunk = text[:remaining]
        parts.append(chunk)
        total += len(chunk)

    return "\n\n".join(parts)
