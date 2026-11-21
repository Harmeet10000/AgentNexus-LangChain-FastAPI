"""Application-owned durable job staging interface for search."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from app.features.search.application.ports.repository import SearchTransaction


class SearchIngestionJob(BaseModel):
    """Search-owned job intent; the adapter selects the stable wire schema."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    tenant_id: str
    content: str
    content_hash: str
    title: str
    source_uri: str | None = None
    doc_metadata: dict[str, object]


class SearchJobQueue(Protocol):
    """Stage search ingestion in the repository's caller-owned transaction."""

    async def stage_ingestion(
        self,
        transaction: SearchTransaction,
        *,
        job: SearchIngestionJob,
    ) -> str: ...
