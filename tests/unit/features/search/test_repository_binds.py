"""asyncpg-safe binds and the documented text() fallbacks, gated for search.

Mirrors ``tests/unit/documents/test_asyncpg_bind_casts.py``: uuid binds must
cross the asyncpg driver as text plus an explicit CAST (never ``:x::uuid``),
``text()`` is reserved for the extension operators and maintenance commands,
and every retrieval branch carries the tenant predicate — compiled or
source-pinned, because these are wire/SQL-shape contracts a live database is
the wrong place to discover.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError

from app.features.search.adapters.persistence.postgres.search_repository import (
    _is_tenant_content_hash_race,
    build_chunk_rows,
    build_chunk_upsert_statement,
)
from app.utils.sql_binds import uuid_any, uuid_param

_REPO_SOURCE = Path(
    "src/app/features/search/adapters/persistence/postgres/search_repository.py"
).read_text(encoding="utf-8")

_DOCUMENT_TENANT = "tenant-1"
_FILTER_SOURCE_TENANT_PREDICATE = "c.tenant_id = :tenant_id"


def test_uuid_param_renders_explicit_cast() -> None:
    rendered = str(
        uuid_param("document_id", "00000000-0000-0000-0000-000000000001").compile(
            dialect=postgresql.dialect()
        )
    )
    assert "CAST(" in rendered
    assert "document_id::uuid" not in rendered


def test_uuid_any_renders_any_over_cast_array() -> None:
    rendered = str(
        uuid_any("chunk_ids", ["00000000-0000-0000-0000-000000000001"]).compile(
            dialect=postgresql.dialect()
        )
    )
    assert "ANY (CAST(" in rendered


def test_repository_uses_the_bind_cast_helpers_not_postfix_casts() -> None:
    assert "uuid_param(" in _REPO_SOURCE
    assert "uuid_any(" in _REPO_SOURCE
    assert "::uuid" not in _REPO_SOURCE.replace("CAST(:chunk_ids AS uuid[])", "")


def test_text_is_reserved_for_extension_operators_and_maintenance() -> None:
    """The sanctioned text() shapes: <@>, <=>, %/similarity + GUCs, ANALYZE."""
    assert _REPO_SOURCE.count("<@> to_bm25query") >= 2
    assert "<=> CAST(:embedding AS vector)" in _REPO_SOURCE
    assert "SET LOCAL diskann.query_search_list_size" in _REPO_SOURCE
    assert "SET LOCAL diskann.query_rescore" in _REPO_SOURCE
    assert "similarity(c.content, :query)" in _REPO_SOURCE
    # The `%` operator reads pg_trgm.similarity_threshold from the session;
    # set_config(..., is_local => true) is its parameterizable SET LOCAL form.
    assert "set_config('pg_trgm.similarity_threshold', :threshold, true)" in _REPO_SOURCE
    assert 'text(text="ANALYZE search_chunks")' in _REPO_SOURCE


def test_every_retrieval_branch_carries_the_tenant_predicate() -> None:
    """The shared filter is appended to bm25, vector, and trigram alike."""
    assert _FILTER_SOURCE_TENANT_PREDICATE in _REPO_SOURCE
    # One appended filter per branch (bm25, vector, trigram) — a branch that
    # interpolates _FILTER_SQL zero or two times fails here.
    assert _REPO_SOURCE.count("+ _FILTER_SQL") == 3


def test_hydration_fetch_is_tenant_scoped_in_the_builder() -> None:
    """fetch_chunks_by_ids pins the tenant in the ORM where clause, not text()."""
    fetch_section = _REPO_SOURCE.split("async def fetch_chunks_by_ids", 1)[1]
    assert "SearchChunk.tenant_id == tenant_id" in fetch_section


def test_chunk_upsert_statement_compiles_with_conflict_refresh_columns() -> None:
    rows = build_chunk_rows(
        document_id="00000000-0000-0000-0000-000000000001",
        tenant_id=_DOCUMENT_TENANT,
        chunks=[
            {
                "id": "00000000-0000-0000-0000-0000000000f1",
                "chunk_index": 0,
                "content": "chunk body",
                "embedding": [0.1, 0.2],
                "chunk_metadata": {"token_count": 2},
            }
        ],
    )
    statement = build_chunk_upsert_statement(rows)
    rendered = str(statement.compile(dialect=postgresql.dialect()))
    statement_text = str(statement)

    assert "ON CONFLICT" in rendered
    assert "uq_search_chunks_document_chunk_index" in statement_text
    # SQLAlchemy does not merge onupdate defaults into DO UPDATE SET, so the
    # refresh columns must be named explicitly — updated_at especially.
    set_clause = statement_text.split("SET ", 1)[1]
    for column in ("content", "embedding", "chunk_metadata", "updated_at"):
        assert f"{column} = " in set_clause


class _FakeUniqueViolationError(Exception):
    sqlstate = "23505"


def _integrity_error(content_hash: str, sqlstate: str | None = "23505") -> IntegrityError:
    orig = _FakeUniqueViolationError()
    if sqlstate is None:
        orig = Exception("driver without diagnostics")
    else:
        _FakeUniqueViolationError.sqlstate = sqlstate  # type: ignore[misc]
    return IntegrityError("INSERT INTO search_documents ...", {"content_hash": content_hash}, orig)


def test_tenant_content_hash_race_is_detected_from_the_constraint() -> None:
    """A concurrent identical ingest losing the unique race is idempotent, not
    an infrastructure error: the winner's row is the dedupe identity."""
    exc = _integrity_error("abc123")

    assert _is_tenant_content_hash_race(exc, content_hash="abc123")


def test_other_unique_violations_are_not_treated_as_the_race() -> None:
    different_hash = _integrity_error("unrelated-value")
    assert not _is_tenant_content_hash_race(different_hash, content_hash="abc123")


def test_non_unique_integrity_errors_are_not_treated_as_the_race() -> None:
    foreign_key = _integrity_error("abc123", sqlstate="23503")
    assert not _is_tenant_content_hash_race(foreign_key, content_hash="abc123")


def test_chunk_rows_carry_the_tenant() -> None:
    rows = build_chunk_rows(
        document_id="00000000-0000-0000-0000-000000000001",
        tenant_id=_DOCUMENT_TENANT,
        chunks=[
            {
                "id": "00000000-0000-0000-0000-0000000000f1",
                "chunk_index": 0,
                "content": "chunk body",
                "embedding": [0.1, 0.2],
                "chunk_metadata": {},
            }
        ],
    )
    assert rows[0]["tenant_id"] == _DOCUMENT_TENANT
