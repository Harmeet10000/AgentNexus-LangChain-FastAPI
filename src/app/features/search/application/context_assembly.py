"""RAG context assembly for site-search hybrid results.

Ported from the legacy ``logs/search/rag.py`` with one modernization: the
token budget is enforced through the injected :data:`app.retrieval.token_counter.CountTokens`
callable (``cl100k_base``) instead of whitespace splitting, so the budget the
endpoint names and the budget the assembler enforces are the same unit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from app.retrieval.token_counter import count_tokens

if TYPE_CHECKING:
    from app.retrieval.fusion import RankedChunk
    from app.retrieval.token_counter import CountTokens


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
    count_tokens: CountTokens = count_tokens,
) -> list[ContextSection]:
    """Group by document, restore chunk order, merge adjacent runs, and budget output.

    The budget splits at chunk granularity: a merged run that would overflow is
    emitted as its largest fitting prefix, and the overflow chunk starts the
    next run if it fits on its own. A group is never dropped wholesale when its
    first chunk fits — only a chunk that cannot fit even alone is skipped.
    """
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
        ordered_chunks = sorted(grouped[document_id], key=lambda item: item[1].chunk_index)
        current_group: list[SearchChunkRecord] = []

        for _, chunk in ordered_chunks:
            if not current_group:
                if used_tokens + count_tokens(chunk.content) <= max_tokens:
                    current_group = [chunk]
                # A chunk that cannot fit even alone is skipped; there is no
                # smaller prefix of it to keep.
                continue

            if chunk.chunk_index == current_group[-1].chunk_index + 1:
                merged = _build_context_section([*current_group, chunk])
                if used_tokens + count_tokens(merged.content) <= max_tokens:
                    current_group.append(chunk)
                    continue
                # Budget boundary inside a merged run: flush the largest
                # fitting prefix, then try to start the next run with chunk.
                used_tokens += _flush(sections, current_group, count_tokens)
                current_group = []
                if used_tokens + count_tokens(chunk.content) <= max_tokens:
                    current_group = [chunk]
                continue

            # Adjacency break: the run is final, and it was admitted under the
            # budget incrementally, so it always fits.
            used_tokens += _flush(sections, current_group, count_tokens)
            current_group = []
            if used_tokens + count_tokens(chunk.content) <= max_tokens:
                current_group = [chunk]

        if current_group:
            used_tokens += _flush(sections, current_group, count_tokens)

    return sections


def _flush(
    sections: list[ContextSection],
    group: list[SearchChunkRecord],
    count_tokens: CountTokens,
) -> int:
    """Emit ``group`` as a section and return its token cost."""
    section = _build_context_section(group)
    sections.append(section)
    return count_tokens(section.content)


def _build_context_section(chunks: list[SearchChunkRecord]) -> ContextSection:
    first_chunk = chunks[0]
    return ContextSection(
        document_id=first_chunk.document_id,
        title=first_chunk.title,
        content="\n\n".join(chunk.content for chunk in chunks),
        chunk_indices=[chunk.chunk_index for chunk in chunks],
        chunk_metadata=first_chunk.chunk_metadata,
    )
