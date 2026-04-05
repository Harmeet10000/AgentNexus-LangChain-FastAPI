"""Token-based chunking helpers for search ingestion."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class TextChunk(BaseModel):
    """Normalized text chunk plus its ordinal position."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_index: int
    content: str
    token_count: int


def chunk_text(text: str, *, chunk_size: int, chunk_overlap: int) -> list[TextChunk]:
    """Split text into overlapping token windows while preserving order."""
    normalized = " ".join(text.split())
    if not normalized:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    tokens = normalized.split(" ")
    step = chunk_size - chunk_overlap
    chunks: list[TextChunk] = []

    for chunk_index, start in enumerate(range(0, len(tokens), step)):
        window = tokens[start : start + chunk_size]
        if not window:
            continue
        chunks.append(
            TextChunk(
                chunk_index=chunk_index,
                content=" ".join(window),
                token_count=len(window),
            )
        )
        if start + chunk_size >= len(tokens):
            break

    return chunks
