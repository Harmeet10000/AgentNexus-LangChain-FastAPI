"""Chunk identity: version coexistence and absent locus (ingestion-chunking 2.3)."""

from __future__ import annotations

from app.features.documents.classification import PreparedChunk
from app.features.documents.constants import CHUNKS_UNIQUE_CONSTRAINT_NAME
from app.features.documents.model import UnifiedChunk
from app.features.documents.repository import build_chunk_rows


def test_unique_constraint_includes_document_version() -> None:
    names = {constraint.name for constraint in UnifiedChunk.__table__.constraints}
    assert CHUNKS_UNIQUE_CONSTRAINT_NAME in names
    assert CHUNKS_UNIQUE_CONSTRAINT_NAME == "uq_chunks_document_version_chunk_index"


def test_two_versions_of_one_document_produce_distinguishable_rows() -> None:
    rows = build_chunk_rows(
        document_id="00000000-0000-0000-0000-000000000001",
        user_id="user-1",
        chunks=[
            {"content": "v1 body", "chunk_index": 0, "document_version": 1},
            {"content": "v2 body", "chunk_index": 0, "document_version": 2},
        ],
    )

    assert {(row["document_version"], row["chunk_index"]) for row in rows} == {(1, 0), (2, 0)}
    assert rows[0]["content"] != rows[1]["content"]


def test_empty_locus_is_stored_as_absent() -> None:
    (row,) = build_chunk_rows(
        document_id="00000000-0000-0000-0000-000000000001",
        user_id="user-1",
        chunks=[{"content": "body", "chunk_index": 0, "locus": ""}],
    )

    assert row["locus"] is None


def test_prepared_chunk_defaults_version_and_absent_locus() -> None:
    chunk = PreparedChunk(
        chunk_index=0,
        chunk_kind="contracts",
        content="body",
    )

    assert chunk.document_version == 1
    assert chunk.locus is None
