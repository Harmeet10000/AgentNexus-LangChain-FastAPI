import asyncio
from collections.abc import AsyncIterator, Iterable, Sequence

import pageindex

from app.config import get_settings
from app.utils import logger

from .client import (
    PageIndexBatchConfig,
    PageIndexChatConfig,
    PageIndexConfig,
    _get_sdk_client,
)


def _raise_non_iterable_stream_response() -> None:
    raise TypeError("Expected iterable response from streamed chat_completions.")


async def apage_index(
    *,
    doc: str | bytes,
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
    except Exception:
        logger.bind(doc=doc, model=runtime.model).exception("PageIndex indexing failed")
        raise
    else:
        return result


async def abatch_page_index(
    *,
    docs: Sequence[str | bytes],
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
    doc_id: str | list[str],
    messages: list[dict[str, str]],
    config: PageIndexChatConfig | None = None,
) -> object:
    """Call PageIndex chat completion API in non-streaming mode."""
    runtime = config or PageIndexChatConfig()
    client = _get_sdk_client()

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
    except Exception:
        logger.bind(doc_id=doc_id).exception("PageIndex chat completion failed")
        raise
    else:
        return result


async def astream_chat_completions(
    *,
    doc_id: str | list[str],
    messages: list[dict[str, str]],
    config: PageIndexChatConfig | None = None,
) -> AsyncIterator[object]:
    """Stream chat completion chunks from PageIndex as an async generator.

    Bridges the SDK's sync iterator to async code using asyncio.to_thread and a queue.
    """
    runtime = config or PageIndexChatConfig(stream=True)
    client = _get_sdk_client()

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
            logger.bind(doc_id=doc_id).exception("PageIndex streaming chat completion failed")
            await queue.put(exc)
        finally:
            await queue.put(end_marker)

    task = asyncio.create_task(_producer())
    try:
        while True:
            item = await queue.get()
            if item is end_marker:
                logger.bind(doc_id=doc_id).info("PageIndex streaming chat completion finished")
                break
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        await task


def create_node_map(tree: dict[str, object]) -> dict[str, dict[str, object]]:
    """Flatten tree for fast node lookup by node_id."""
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
    """Extract clean text from selected nodes for LangGraph workflows."""
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
