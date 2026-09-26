"""RAG assembly helpers for hybrid search results."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from app.shared.rag.token_counter import CountTokens

    from .fusion import RankedChunk


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
    chunk_ids: list[str]
    chunk_indices: list[int]
    chunk_metadata: dict[str, object]


def assemble_rag_context(
    ranked_chunks: list[RankedChunk],
    chunk_lookup: dict[str, SearchChunkRecord],
    *,
    max_tokens: int,
    count_tokens: CountTokens,
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
        for group in _adjacent_groups(ordered_chunks):
            offset = 0
            while offset < len(group):
                remaining_tokens = max_tokens - used_tokens
                if remaining_tokens <= 0:
                    return sections

                prefix_length, section, section_tokens = _largest_fitting_prefix(
                    group[offset:],
                    max_tokens=remaining_tokens,
                    count_tokens=count_tokens,
                )
                if prefix_length == 0:
                    # An oversized chunk must not prevent smaller, later
                    # evidence in the same source document from being used.
                    offset += 1
                    continue

                sections.append(section)
                used_tokens += section_tokens
                offset += prefix_length

    return sections


def _adjacent_groups(
    chunks: list[tuple[RankedChunk, SearchChunkRecord]],
) -> list[list[tuple[RankedChunk, SearchChunkRecord]]]:
    """Partition ordered chunks into maximal groups with consecutive indexes."""
    groups: list[list[tuple[RankedChunk, SearchChunkRecord]]] = []
    for pair in chunks:
        if groups and pair[1].chunk_index == groups[-1][-1][1].chunk_index + 1:
            groups[-1].append(pair)
        else:
            groups.append([pair])
    return groups


def _largest_fitting_prefix(
    chunks: list[tuple[RankedChunk, SearchChunkRecord]],
    *,
    max_tokens: int,
    count_tokens: CountTokens,
) -> tuple[int, ContextSection, int]:
    """Return the longest prefix within the budget using logarithmic token counts.

    Most adjacent groups fit and require a single tokenizer call. Oversized groups
    use a binary search rather than re-tokenizing every growing prefix.
    """
    full_section = _build_context_section(chunks)
    full_tokens = count_tokens(full_section.content)
    if full_tokens <= max_tokens:
        return len(chunks), full_section, full_tokens

    low = 1
    high = len(chunks) - 1
    best_length = 0
    best_section = full_section
    best_tokens = full_tokens

    while low <= high:
        middle = (low + high) // 2
        candidate = _build_context_section(chunks[:middle])
        candidate_tokens = count_tokens(candidate.content)
        if candidate_tokens <= max_tokens:
            best_length = middle
            best_section = candidate
            best_tokens = candidate_tokens
            low = middle + 1
        else:
            high = middle - 1

    return best_length, best_section, best_tokens


def _build_context_section(
    chunks: list[tuple[RankedChunk, SearchChunkRecord]],
) -> ContextSection:
    first_chunk = chunks[0][1]
    return ContextSection(
        document_id=first_chunk.document_id,
        title=first_chunk.title,
        content="\n\n".join(chunk.content for _, chunk in chunks),
        chunk_ids=[ranked_chunk.chunk_id for ranked_chunk, _ in chunks],
        chunk_indices=[chunk.chunk_index for _, chunk in chunks],
        chunk_metadata=first_chunk.chunk_metadata,
    )
