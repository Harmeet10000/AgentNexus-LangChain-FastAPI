# src/app/shared/rag/pageindex/client.py
"""PageIndex integration - thin SDK wrapper + async safety layer."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import pageindex
from asyncer import asyncify
from pydantic import BaseModel, Field
from returns.result import Failure, Success

from app.config import get_settings
from app.shared.rag.errors import RagServiceError, RagValidationError
from app.utils import logger

if TYPE_CHECKING:
    from app.shared.rag.errors import RagResult


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


@cache
def _get_sdk_client() -> RagResult[pageindex.PageIndexClient]:
    """Cached SDK client (handles pooling internally)."""
    settings = get_settings()  # or get_settings()
    if not settings.PAGEINDEX_API_KEY.get_secret_value():
        msg = "PAGEINDEX_API_KEY is required"
        return Failure(
            RagValidationError(
                message=msg,
                source="pageindex_client",
                details={"operation": "_get_sdk_client"},
            )
        )
    return Success(pageindex.PageIndexClient(api_key=settings.PAGEINDEX_API_KEY.get_secret_value()))


class PageIndexClient:
    """Main injectable client. Lives in app.state."""

    def __init__(self) -> None:
        self._sdk_result = _get_sdk_client()

    # === Core methods (thin wrappers) ===

    async def submit_document(self, file_path: str | bytes) -> RagResult[str]:
        """Submit for indexing. Prefer Celery for production."""
        sdk_result = self._sdk_result
        if isinstance(sdk_result, Failure):
            return Failure(sdk_result.failure())
        sdk = sdk_result.unwrap()
        try:
            result = await asyncify(sdk.submit_document)(file_path)
            doc_id: str = result["doc_id"]
            logger.bind(doc_id=doc_id).info("document_submitted")
        except Exception as exc:  # noqa: BLE001 — PageIndex SDK, unknown failure modes
            exc.add_note("operation=submit_document")
            logger.bind(operation="submit_document", error=str(exc)).exception("submit_failed")
            return Failure(
                RagServiceError(
                    message="PageIndex submit failed",
                    source="pageindex_client",
                    operation="submit_document",
                    details={"notes": list(getattr(exc, "__notes__", []))},
                )
            )
        else:
            return Success(doc_id)

    async def get_tree(self, doc_id: str, node_summary: bool = True) -> RagResult[dict[str, object]]:
        sdk_result = self._sdk_result
        if isinstance(sdk_result, Failure):
            return Failure(sdk_result.failure())
        sdk = sdk_result.unwrap()
        result = await asyncify(sdk.get_tree)(doc_id, node_summary=node_summary)
        return Success(result.get("result", {}))

    # ... add get_document_status, etc. as needed
