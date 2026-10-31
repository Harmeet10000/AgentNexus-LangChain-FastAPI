"""Ported chunker contract: overlapping token windows, order-stable indexes.

Site search keeps the legacy whitespace-token windower (the docling chunker is
document-pipeline-specific), so the ingest pipeline's idempotency claim —
identical content produces identical chunk indexes — rests on this contract.
"""

from __future__ import annotations

import pytest

from app.features.search.domain.chunking import chunk_text


def test_empty_text_yields_no_chunks() -> None:
    assert chunk_text("", chunk_size=8, chunk_overlap=2) == []
    assert chunk_text("   \n\t  ", chunk_size=8, chunk_overlap=2) == []


def test_short_text_is_a_single_chunk() -> None:
    chunks = chunk_text("one two three", chunk_size=8, chunk_overlap=2)
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].content == "one two three"
    assert chunks[0].token_count == 3


def test_windows_overlap_by_the_configured_step() -> None:
    tokens = [str(index) for index in range(10)]
    chunks = chunk_text(" ".join(tokens), chunk_size=4, chunk_overlap=1)
    # step = 4 - 1 = 3 → starts at 0, 3, 6; the final window reaching token 9
    # ends the scan (`start + chunk_size >= len(tokens)` breaks the loop), so
    # the tail is never re-emitted as a lone chunk.
    assert [chunk.content for chunk in chunks] == [
        "0 1 2 3",
        "3 4 5 6",
        "6 7 8 9",
    ]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]


def test_identical_content_produces_identical_chunks() -> None:
    content = " ".join(f"token{index}" for index in range(2000))
    first = chunk_text(content, chunk_size=512, chunk_overlap=64)
    second = chunk_text(content, chunk_size=512, chunk_overlap=64)
    assert first == second


def test_whitespace_drift_is_normalized() -> None:
    assert chunk_text("a\n\nb\t c", chunk_size=8, chunk_overlap=0) == chunk_text(
        "a b c", chunk_size=8, chunk_overlap=0
    )


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(0, 0), (-1, 0), (8, 8), (8, 9), (8, -1)],
)
def test_invalid_geometry_is_rejected(chunk_size: int, chunk_overlap: int) -> None:
    with pytest.raises(ValueError, match="chunk_"):
        chunk_text("a b c", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
