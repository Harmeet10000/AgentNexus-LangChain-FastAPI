"""RAG assembly helpers for hybrid search results."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from app.features.search.fusion import RankedChunk


class SearchChunkRecord(BaseModel):
    """Hydrated chunk record used for search response assembly."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    title: str
    content: str
    chunk_index: int
    chunk_metadata: dict[str, object]


class ContextSection(BaseModel):
    """Merged context section returned to RAG callers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    title: str
    content: str
    chunk_indices: list[int]
    chunk_metadata: dict[str, object]


def assemble_rag_context(
    ranked_chunks: list[RankedChunk],
    chunk_lookup: dict[str, SearchChunkRecord],
    *,
    max_tokens: int,
) -> list[ContextSection]:
    """Group by document, restore chunk order, merge adjacent chunks, and budget output."""
    grouped: dict[str, list[tuple[RankedChunk, SearchChunkRecord]]] = {}
    document_order: list[str] = []

    for ranked_chunk in ranked_chunks:
        chunk = chunk_lookup.get(ranked_chunk.chunk_id)
        if chunk is None:
            continue
        document_id = chunk.document_id
        if document_id not in grouped:
            grouped[document_id] = []
            document_order.append(document_id)
        grouped[document_id].append((ranked_chunk, chunk))

    sections: list[ContextSection] = []
    used_tokens = 0

    for document_id in document_order:
        ordered_chunks = sorted(
            grouped[document_id],
            key=lambda item: item[1].chunk_index,
        )
        current_group: list[SearchChunkRecord] = []

        for _, chunk in ordered_chunks:
            if not current_group:
                current_group = [chunk]
                continue

            previous_index = current_group[-1].chunk_index
            current_index = chunk.chunk_index
            if current_index == previous_index + 1:
                current_group.append(chunk)
                continue

            section = _build_context_section(current_group)
            used_tokens += len(section.content.split())
            if used_tokens > max_tokens:
                return sections
            sections.append(section)
            current_group = [chunk]

        if current_group:
            section = _build_context_section(current_group)
            used_tokens += len(section.content.split())
            if used_tokens > max_tokens:
                return sections
            sections.append(section)

    return sections


def _build_context_section(chunks: list[SearchChunkRecord]) -> ContextSection:
    first_chunk = chunks[0]
    return ContextSection(
        document_id=first_chunk.document_id,
        title=first_chunk.title,
        content="\n\n".join(chunk.content for chunk in chunks),
        chunk_indices=[chunk.chunk_index for chunk in chunks],
        chunk_metadata=first_chunk.chunk_metadata,
    )
