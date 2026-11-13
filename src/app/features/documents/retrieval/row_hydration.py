"""The single chunk-row hydration path (harden 5.3).

Every retrieval surface — the compiled graph's hybrid node, the shared
`retrieve(...)` service, and the evidence contract — hydrates persisted chunk
rows through this module and only through this module. Provenance reading
(document version, pipeline version, contract index, graph-indexed timestamp,
page number) and the `RetrievedChunk` projection used to live as two private
helper sets (`graph.nodes._row_to_chunk` and `application.evidence.service._*`);
duplicates drift, so both now import from here. Field semantics are unchanged:
this is a consolidation, not a contract change.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel, ConfigDict

from app.features.documents.retrieval.graph.state import RetrievedChunk

if TYPE_CHECKING:
    from typing import Any


class HydratedChunk(BaseModel):
    """The retrieval DTO's provenance fields (harden 5.3 / retrieval schema).

    Field-identical provenance to `GroundedCitation`/`EvidenceRef`: page
    number, document version, pipeline version, contract index, and the graph
    coverage signal. ``None`` pipeline/contract values mean legacy rows and
    are never presented as graph-verified.
    """

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    page_no: int = 0
    document_version: int = 1
    pipeline_version: str | None = None
    contract_index: int | None = None
    graph_indexed: bool = False
    graph_indexed_at: datetime | None = None


def document_version(row: dict[str, object]) -> int:
    value = row.get("document_version", 1)
    try:
        return int(cast("int | str", value))
    except (TypeError, ValueError):
        return 1


def pipeline_version(row: dict[str, object]) -> str | None:
    explicit = row.get("pipeline_version")
    if explicit is not None:
        return str(explicit)
    for key in ("chunk_metadata", "document_metadata", "metadata_"):
        value = row.get(key)
        if isinstance(value, dict):
            version = value.get("pipeline_version") or value.get("extraction_version")
            if version is not None:
                return str(version)
    return None


def contract_index(row: dict[str, object]) -> int | None:
    """Read the explicit multi-contract identity, preserving legacy NULL."""
    value = row.get("contract_index")
    if value is None:
        value = record_metadata(row).get("contract_index")
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str):
        try:
            parsed = int(value)
        except ValueError:
            return None
        return parsed if parsed >= 0 else None
    return None


def record_metadata(row: dict[str, object]) -> dict[str, object]:
    for key in ("chunk_metadata", "metadata_"):
        value = row.get(key)
        if isinstance(value, dict):
            return {str(item_key): item for item_key, item in value.items()}
    return {}


def graph_indexed_at(row: dict[str, object]) -> datetime | None:
    value = row.get("graph_indexed_at")
    return value if isinstance(value, datetime) else None


def graph_indexed(row: dict[str, object]) -> bool:
    return bool(row.get("graphiti_verified"))


def page_no(row: dict[str, object]) -> int:
    value = row.get("page_no")
    try:
        return int(cast("int | str", value)) if value is not None else 0
    except (TypeError, ValueError):
        return 0


def provenance(row: dict[str, object]) -> HydratedChunk:
    """The provenance projection of one hydrated row (harden 5.3)."""
    return HydratedChunk(
        chunk_id=str(row.get("chunk_id", "")),
        document_id=str(row.get("document_id", "")),
        page_no=page_no(row),
        document_version=document_version(row),
        pipeline_version=pipeline_version(row),
        contract_index=contract_index(row),
        graph_indexed=graph_indexed(row),
        graph_indexed_at=graph_indexed_at(row),
    )


def bounded_snippet(content: str, maximum: int) -> str:
    if len(content) <= maximum:
        return content
    return content[: max(1, maximum - 1)].rstrip() + "…"


def row_to_chunk(row: dict[str, Any]) -> RetrievedChunk:
    """Project a fused, hydrated row into the graph's `RetrievedChunk`.

    Moved verbatim from `graph.nodes._row_to_chunk` (the inline contract the
    parity suite pins): the contract index and graph-indexing state ride in
    metadata so the chunk payloads the grader and generator consume stay
    byte-identical across both paths.
    """
    chunk_index = row.get("chunk_index")
    return RetrievedChunk(
        chunk_id=str(row["chunk_id"]),
        chunk_text=str(row["chunk_text"]),
        preamble=str(row["preamble"] or ""),
        clause_type=str(row["clause_type"]),
        parent_doc_id=str(row["parent_doc_id"]),
        chunk_index=int(chunk_index) if chunk_index is not None else 0,
        # The contract index is never populated on the field the LLM could
        # see — it rides in metadata, exactly as the inline `_row_to_chunk`
        # leaves it — so the chunk payloads the grader and generator consume
        # are byte-identical across both paths.
        # The same server-metadata enrichment the inline `_row_to_chunk`
        # performs: quality warnings, graph-indexing state, document version,
        # pipeline version, and contract index ride in metadata so edge-seed
        # resolution and citation provenance read them from one place.
        metadata_={
            **dict(row["metadata_"] or {}),
            "quality_warnings": row.get("quality_warnings", []),
            "graphiti_verified": bool(row.get("graphiti_verified")),
            "graph_indexed": bool(row.get("graphiti_verified")),
            "document_version": row.get("document_version", 1),
            "pipeline_version": row.get("pipeline_version"),
            "contract_index": row.get("contract_index"),
            "graph_indexed_at": row.get("graph_indexed_at"),
        },
        custom_metadata=dict(row["custom_metadata"] or {}),
        score=float(row["rrf_score"]),
    )
