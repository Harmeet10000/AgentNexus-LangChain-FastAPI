"""STRIDE D-3: the search twin keeps the quota cause typed at its boundary.

The search pipeline's ``embed_texts`` call sits inside the task body, whose
``dont_autoretry_for`` exempts the terminal class. A daily-quota 429 must
arrive there as ``SearchIngestQuotaExhaustedError`` — dead-lettering
immediately with the typed code — never as a raw provider error and never as
a retry burn.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import app.features.search.application.ingestion as search_ingestion
from app.ai.embeddings import EmbeddingQuotaExhaustedError
from app.features.search.application.ingestion import (
    SearchIngestQuotaExhaustedError,
    SearchIngestTerminalError,
    process_search_ingestion_document,
)

if TYPE_CHECKING:
    from typing import Any

_CONTENT = ("Clause text for the search chunking pipeline. " * 40)


class _UnusedRepo:
    def __getattr__(self, name: str) -> Any:  # pragma: no cover — must not be reached
        violation = f"repo.{name} reached before the embedding failure"
        raise AssertionError(violation)


async def test_quota_failure_raises_typed_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _quota_boom(*_args: Any, **_kwargs: Any) -> list[list[float]]:
        quota_detail = (
            "Embedding provider daily quota exhausted for model gemini-embedding-001"
        )
        raise EmbeddingQuotaExhaustedError(
            quota_detail,
            model="gemini-embedding-001",
            provider_status=429,
        )

    monkeypatch.setattr(search_ingestion, "embed_texts", _quota_boom)

    with pytest.raises(SearchIngestQuotaExhaustedError) as raised:
        await process_search_ingestion_document(
            repo=_UnusedRepo(),  # type: ignore[arg-type]
            document_id="doc-1",
            tenant_id="tenant-1",
            content=_CONTENT,
        )

    assert raised.value.error_code == "embedding_quota_exhausted"
    assert isinstance(raised.value.__cause__, EmbeddingQuotaExhaustedError)


def test_quota_error_is_terminal_for_the_autoretry_policy() -> None:
    # The task's dont_autoretry_for names the terminal base; the quota error
    # must be covered by that exemption, not sit beside it.
    assert issubclass(SearchIngestQuotaExhaustedError, SearchIngestTerminalError)
