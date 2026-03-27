"""Async helpers for PageIndex with concurrency and backpressure controls."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable, Sequence
from functools import lru_cache

import pageindex
from pydantic import BaseModel, Field

from app.config.settings import get_settings
from app.utils import logger


class PageIndexConfig(BaseModel):
    """Configuration for indexing operations."""

    model_config = {"frozen": True, "extra": "forbid"}

    api_key: str | None = None
    model: str = "gpt-4o-2024-11-20"
    toc_check_page_num: int = 20
    max_page_num_each_node: int = 10
    max_token_num_each_node: int = 20_000
    if_add_node_id: str = "yes"
    if_add_node_summary: str = "yes"
    if_add_doc_description: str = "no"
    if_add_node_text: str = "no"
    additional_kwargs: dict[str, object] = Field(default_factory=dict)


class PageIndexBatchConfig(BaseModel):
    """Concurrency settings for indexing batches."""

    model_config = {"frozen": True, "extra": "forbid"}

    max_concurrency: int = 4


class PageIndexChatConfig(BaseModel):
    """Configuration for PageIndex chat completion calls."""

    model_config = {"frozen": True, "extra": "forbid"}

    api_key: str | None = None
    model: str | None = None
    stream: bool = False
    temperature: float | None = None
    additional_kwargs: dict[str, object] = Field(default_factory=dict)


@lru_cache(maxsize=1)
def _get_client() -> object:
    """Get or create a cached PageIndex client instance.

    The SDK handles connection pooling internally, so we only need one client instance.
    """
    settings = get_settings()
    api_key = settings.PAGEINDEX_API_KEY
    if not api_key:
        msg = "PAGEINDEX_API_KEY is required for PageIndex operations."
        logger.error(msg)
        raise ValueError(msg)
    return pageindex.PageIndexClient(api_key=api_key)


def _raise_non_iterable_stream_response() -> None:
    raise TypeError("Expected iterable response from streamed chat_completions.")


async def apage_index(
    *,
    doc: str,
    config: PageIndexConfig | None = None,
) -> dict[str, object]:
    """Run PageIndex document indexing without blocking the event loop."""
    runtime = config or PageIndexConfig()
    settings = get_settings()
    api_key = runtime.api_key or settings.PAGEINDEX_API_KEY
    if not api_key:
        msg = "PAGEINDEX_API_KEY is required for PageIndex operations."
        logger.error(msg)
        raise ValueError(msg)

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
        "api_key": api_key,
    }
    if runtime.additional_kwargs:
        kwargs.update(runtime.additional_kwargs)

    logger.bind(doc=doc, model=runtime.model).info("Starting PageIndex indexing")
    try:
        result = await asyncio.to_thread(pageindex.page_index, **kwargs)
        logger.bind(doc=doc).info("PageIndex indexing completed successfully")
        return result
    except Exception:
        logger.bind(doc=doc, model=runtime.model).exception(
            "PageIndex indexing failed"
        )
        raise


async def abatch_page_index(
    *,
    docs: Sequence[str],
    config: PageIndexConfig | None = None,
    batch_config: PageIndexBatchConfig | None = None,
) -> list[dict[str, object]]:
    """Index multiple documents concurrently with a bounded in-flight window.

    Raises exceptions immediately if any task fails.
    """
    if not docs:
        return []

    runtime_config = config or PageIndexConfig()
    runtime_batch = batch_config or PageIndexBatchConfig()
    semaphore = asyncio.Semaphore(max(1, runtime_batch.max_concurrency))

    logger.bind(doc_count=len(docs), max_concurrency=runtime_batch.max_concurrency).info(
        "Starting batch PageIndex indexing"
    )

    async def _worker(doc_path: str) -> dict[str, object]:
        async with semaphore:
            return await apage_index(doc=doc_path, config=runtime_config)

    tasks = [asyncio.create_task(_worker(doc_path)) for doc_path in docs]
    results = await asyncio.gather(*tasks)
    
    logger.bind(doc_count=len(docs), success_count=len(results)).info(
        "Batch PageIndex indexing completed"
    )
    return results


async def achat_completion(
    *,
    doc_id: str,
    messages: list[dict[str, str]],
    config: PageIndexChatConfig | None = None,
) -> object:
    """Call PageIndex chat completion API in non-streaming mode."""
    runtime = config or PageIndexChatConfig()
    client = _get_client()

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

    logger.bind(doc_id=doc_id, message_count=len(messages)).info(
        "Starting PageIndex chat completion"
    )
    try:
        result = await asyncio.to_thread(client.chat_completions, **kwargs)
        logger.bind(doc_id=doc_id).info("PageIndex chat completion successful")
        return result
    except Exception:
        logger.bind(doc_id=doc_id).exception("PageIndex chat completion failed")
        raise


async def astream_chat_completions(
    *,
    doc_id: str,
    messages: list[dict[str, str]],
    config: PageIndexChatConfig | None = None,
) -> AsyncIterator[object]:
    """Stream chat completion chunks from PageIndex as an async generator.

    Bridges the SDK's sync iterator to async code using asyncio.to_thread and a queue.
    """
    runtime = config or PageIndexChatConfig(stream=True)
    client = _get_client()

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

    logger.bind(doc_id=doc_id, message_count=len(messages)).info(
        "Starting PageIndex streaming chat completion"
    )

    queue: asyncio.Queue[object] = asyncio.Queue(maxsize=256)
    end_marker = object()

    async def _producer() -> None:
        try:
            response = await asyncio.to_thread(client.chat_completions, **kwargs)
            if not isinstance(response, Iterable):
                _raise_non_iterable_stream_response()
            for chunk in response:
                await queue.put(chunk)
        except Exception as exc:
            logger.bind(doc_id=doc_id).exception(
                "PageIndex streaming chat completion failed"
            )
            await queue.put(exc)
        finally:
            await queue.put(end_marker)

    task = asyncio.create_task(_producer())
    try:
        while True:
            item = await queue.get()
            if item is end_marker:
                logger.bind(doc_id=doc_id).info(
                    "PageIndex streaming chat completion finished"
                )
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
