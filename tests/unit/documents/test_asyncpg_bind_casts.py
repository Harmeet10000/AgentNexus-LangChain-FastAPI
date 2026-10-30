"""asyncpg-safe binds: uuid values must cross as text plus an explicit CAST.

asyncpg prepares statements and type-checks every parameter, so a bare string
reaching a uuid parameter is rejected at encode time — and a Postgres-style
``:x::uuid`` postfix cannot be parsed on a SQLAlchemy bind placeholder at all.
The single cast idiom lives in ``app/utils/sql_binds.py``; this gate compiles
those helpers against the PostgreSQL dialect and pins both the CAST rendering
and the ``= ANY(...)`` array shape, so a regression back to raw string binds
fails here instead of at the driver.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy.dialects import postgresql

from app.utils.sql_binds import uuid_any, uuid_param

_REPO_SOURCE = Path(
    "src/app/features/documents/adapters/persistence/postgres/document_repository.py"
).read_text(encoding="utf-8")


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
        uuid_any("document_ids", ["00000000-0000-0000-0000-000000000001"]).compile(
            dialect=postgresql.dialect()
        )
    )
    assert "ANY (CAST(" in rendered


def test_repository_uses_the_bind_cast_helpers_not_postfix_casts() -> None:
    assert _REPO_SOURCE.count("uuid_param(") >= 2
    assert "uuid_any(" in _REPO_SOURCE
    assert ":document_id::uuid" not in _REPO_SOURCE
