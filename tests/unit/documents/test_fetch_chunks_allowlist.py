"""Fail-closed allowlist semantics for ``fetch_chunks_by_ids``.

The defect this pins: the ``allowed_doc_ids`` guard was falsy-checked, so an
empty allowlist (an explicit empty scope) skipped the allowlist predicate
entirely and matched every document.  Only ``None`` may omit the predicate; an
empty sequence must keep it and therefore match nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from returns.result import Success
from sqlalchemy.dialects import postgresql

from app.features.documents.adapters.persistence.postgres.document_repository import (
    DocumentRepository,
)

if TYPE_CHECKING:
    from typing import Any

    from sqlalchemy import Select


class _FakeResult:
    def mappings(self) -> _FakeResult:
        return self

    def all(self) -> list[dict[str, object]]:
        return []


class _StatementCaptureSession:
    def __init__(self) -> None:
        self.statements: list[Select] = []

    async def execute(self, statement: Select, *_args: object) -> _FakeResult:
        self.statements.append(statement)
        return _FakeResult()


async def _captured_sql(allowed_doc_ids: tuple[str, ...] | None) -> str:
    session = _StatementCaptureSession()
    repo = DocumentRepository(cast("Any", session))
    result = await repo.fetch_chunks_by_ids(
        ["00000000-0000-0000-0000-00000000000a"],
        allowed_doc_ids=allowed_doc_ids,
    )
    assert isinstance(result, Success)
    assert result.unwrap() == {}
    statement = session.statements[-1]
    return str(statement.compile(dialect=postgresql.dialect()))


async def test_empty_allowlist_keeps_the_predicate_and_matches_nothing() -> None:
    sql = await _captured_sql(())

    assert "allowed_doc_ids" in sql
    assert "ANY" in sql


async def test_none_allowlist_omits_the_predicate() -> None:
    sql = await _captured_sql(None)

    assert "allowed_doc_ids" not in sql
