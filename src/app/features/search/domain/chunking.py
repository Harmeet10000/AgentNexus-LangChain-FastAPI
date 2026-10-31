"""Token-based chunking for search ingestion.

Ported from the legacy ``logs/search/chunking.py``: site search ingests
arbitrary text, so it keeps the simple whitespace-token windower rather than
the document-pipeline-specific docling chunker used by ``features/documents``.
"""

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
        msg = "chunk_size must be greater than zero"
        raise ValueError(msg)
    if chunk_overlap < 0:
        msg_overlap = "chunk_overlap must be non-negative"
        raise ValueError(msg_overlap)
    if chunk_overlap >= chunk_size:
        msg_window = "chunk_overlap must be smaller than chunk_size"
        raise ValueError(msg_window)

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
