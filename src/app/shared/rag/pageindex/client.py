# src/app/shared/rag/pageindex/client.py
"""PageIndex integration - thin SDK wrapper + async safety layer."""

from __future__ import annotations

from functools import lru_cache

import pageindex
from asyncer import asyncify
from pydantic import BaseModel, Field

from app.config import get_settings
from app.utils import (
    APIException,
    ValidationException,
    logger,
)


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
    """Concurrency settings for batch indexing."""

    model_config = {"frozen": True, "extra": "forbid"}
    max_concurrency: int = 4


class PageIndexChatConfig(BaseModel):
    """Configuration for chat completion calls."""

    model_config = {"frozen": True, "extra": "forbid"}

    api_key: str | None = None
    model: str | None = None
    stream: bool = False
    temperature: float | None = None
    additional_kwargs: dict[str, object] = Field(default_factory=dict)


@lru_cache(maxsize=1)
def _get_sdk_client() -> pageindex.PageIndexClient:
    """Cached SDK client (handles pooling internally)."""
    settings = get_settings()  # or get_settings()
    if not settings.PAGEINDEX_API_KEY:
        raise ValidationException("PAGEINDEX_API_KEY is required")
    return pageindex.PageIndexClient(api_key=settings.PAGEINDEX_API_KEY)


class PageIndexClient:
    """Main injectable client. Lives in app.state."""

    def __init__(self) -> None:
        self._sdk = _get_sdk_client()

    # === Core methods (thin wrappers) ===

    async def submit_document(self, file_path: str | bytes) -> str:
        """Submit for indexing. Prefer Celery for production."""
        try:
            result = await asyncify(self._sdk.submit_document)(file_path)
            doc_id: str = result["doc_id"]
            logger.info("document_submitted", doc_id=doc_id)
        except Exception as exc:
            logger.exception("submit_failed")
            raise APIException("PageIndex submit failed") from exc
        else:
            return doc_id

    async def get_tree(self, doc_id: str, node_summary: bool = True) -> dict:
        result = await asyncify(self._sdk.get_tree)(doc_id, node_summary=node_summary)
        return result.get("result", {})

    # ... add get_document_status, etc. as needed
