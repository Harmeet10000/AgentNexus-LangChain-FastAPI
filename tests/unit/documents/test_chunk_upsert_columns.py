"""Band D15: the chunk upsert must refresh every mutable column, including updated_at.

The defect this pins: SQLAlchemy applies ``onupdate`` to generated UPDATE
statements but not to an explicit ``DO UPDATE SET``, and chunks have exactly one
write path — that upsert. A column missing from the conflict set is populated on
first insert and then never changes, indistinguishable from a working column
until someone diffs it against ``created_at``.

Both halves are required: the rendered SQL proves the conflict set carries the
names, the row builder proves every write path supplies them.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy.dialects import postgresql

from app.features.documents.repository import build_chunk_rows, build_chunk_upsert_statement

_CONFLICT_SET_COLUMNS = (
    "chunk_kind",
    "content",
    "preamble",
    "clause_type",
    "page_no",
    "embedding",
    "metadata_",
    "custom_metadata",
    "quality_warnings",
    "graphiti_episode_id",
    "graphiti_verified",
    "updated_at",
    "instrument_name",
    "section_ref",
    "instrument_year",
)


def _rendered_upsert_sql(rows: list[dict[str, object]]) -> str:
    statement = build_chunk_upsert_statement(rows)
    return str(statement.compile(dialect=postgresql.dialect()))


def test_conflict_set_refreshes_every_mutable_column() -> None:
    rows = build_chunk_rows(
        document_id="00000000-0000-0000-0000-000000000001",
        user_id="user-1",
        chunks=[{"content": "body", "chunk_index": 0}],
    )
    sql = _rendered_upsert_sql(rows)

    assert "DO UPDATE SET" in sql, "upsert must resolve conflicts with an explicit SET"
    after = sql.split("DO UPDATE SET", 1)[1]
    for name in _CONFLICT_SET_COLUMNS:
        assert name in after, (
            f"{name} is absent from DO UPDATE SET — it would be written once and never refreshed"
        )


def test_build_chunk_rows_carries_updated_at() -> None:
    rows = build_chunk_rows(
        document_id="00000000-0000-0000-0000-000000000001",
        user_id="user-1",
        chunks=[{"content": "body", "chunk_index": 0}, {"content": "more", "chunk_index": 1}],
    )

    for row in rows:
        assert isinstance(row["updated_at"], datetime), (
            "every chunk row must carry updated_at — it is the value the conflict set refreshes with"
        )


def test_build_chunk_rows_honours_caller_supplied_updated_at() -> None:
    pinned = datetime(2026, 1, 1, tzinfo=UTC)
    (row,) = build_chunk_rows(
        document_id="00000000-0000-0000-0000-000000000001",
        user_id="user-1",
        chunks=[{"content": "body", "chunk_index": 0, "updated_at": pinned}],
    )

    assert row["updated_at"] == pinned
