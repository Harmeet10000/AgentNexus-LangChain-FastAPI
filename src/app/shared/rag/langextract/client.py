"""Async helpers for Google LangExtract with performance-oriented defaults."""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING

import langextract as lx

from app.utils import logger

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any


@dataclass(slots=True, frozen=True)
class LangExtractConfig:
    """Config for a single LangExtract invocation."""

    prompt_description: str
    examples: Sequence[Any] = field(default_factory=tuple)
    model_id: str = "gemini-2.5-flash"
    api_key: str | None = None
    extraction_passes: int = 1
    max_char_buffer: int = 4000
    batch_length: int = 10
    max_workers: int | None = None
    fence_output: bool | None = None
    use_schema_constraints: bool | None = None
    language_model_params: dict[str, Any] | None = None
    additional_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class LangExtractBatchConfig:
    """Concurrency settings for async batch extraction."""

    max_concurrency: int = 8
    return_exceptions: bool = True


@lru_cache(maxsize=1)
def _executor() -> ThreadPoolExecutor:
    """Create and cache a shared pool used to offload blocking extract calls."""
    default_workers = max((os.cpu_count() or 4) * 2, 8)
    return ThreadPoolExecutor(max_workers=default_workers, thread_name_prefix="lx")


def _build_extract_kwargs(
    *,
    text_or_documents: str | list[str],
    config: LangExtractConfig,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "text_or_documents": text_or_documents,
        "prompt_description": config.prompt_description,
        "examples": list(config.examples),
        "model_id": config.model_id,
        "extraction_passes": config.extraction_passes,
        "max_char_buffer": config.max_char_buffer,
        "batch_length": config.batch_length,
    }

    if config.api_key:
        kwargs["api_key"] = config.api_key
    if config.max_workers is not None:
        kwargs["max_workers"] = config.max_workers
    if config.fence_output is not None:
        kwargs["fence_output"] = config.fence_output
    if config.use_schema_constraints is not None:
        kwargs["use_schema_constraints"] = config.use_schema_constraints
    if config.language_model_params:
        kwargs["language_model_params"] = config.language_model_params
    if config.additional_kwargs:
        kwargs.update(config.additional_kwargs)

    return kwargs


async def aextract_langextract(
    *,
    text_or_documents: str | list[str],
    config: LangExtractConfig,
) -> Any:
    """
    Async wrapper around `langextract.extract`.

    Performance notes:
    - Offloads sync SDK call to a shared thread pool.
    - Reuses the same pool across requests to avoid executor churn.
    """
    if lx is None:
        raise RuntimeError("langextract is not installed. Add it to runtime dependencies first.")

    loop = asyncio.get_running_loop()
    kwargs = _build_extract_kwargs(text_or_documents=text_or_documents, config=config)

    logger.info(
        "Starting LangExtract call",
        model_id=config.model_id,
        extraction_passes=config.extraction_passes,
    )

    return await loop.run_in_executor(
        _executor(),
        lambda: lx.extract(**kwargs),
    )


async def abatch_extract_langextract(
    *,
    texts_or_documents: Sequence[str | list[str]],
    config: LangExtractConfig,
    batch_config: LangExtractBatchConfig | None = None,
) -> list[Any]:
    """
    Run multiple LangExtract tasks concurrently with backpressure.

    Performance notes:
    - Uses semaphore to cap in-flight tasks and protect API + CPU.
    - Uses a single shared executor for all blocking SDK calls.
    """
    if not texts_or_documents:
        return []

    runtime = batch_config or LangExtractBatchConfig()
    semaphore = asyncio.Semaphore(max(1, runtime.max_concurrency))

    async def _worker(item: str | list[str]) -> Any:
        async with semaphore:
            return await aextract_langextract(text_or_documents=item, config=config)

    tasks = [asyncio.create_task(_worker(item)) for item in texts_or_documents]
    return await asyncio.gather(
        *tasks,
        return_exceptions=runtime.return_exceptions,
    )
