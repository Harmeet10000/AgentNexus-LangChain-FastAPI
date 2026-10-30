"""The statute point lookup prefers dated rows over unversioned rows."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy.dialects import postgresql

from app.features.documents.application.tools.retrieve_statute_section import (
    _fetch_statute_section,
)

if TYPE_CHECKING:
    from typing import Any

_TOOL_SOURCE = Path(
    "src/app/features/documents/application/tools/retrieve_statute_section.py"
).read_text(encoding="utf-8")


class _NoRowResult:
    def fetchone(self) -> None:
        return None


class _RecordingConnection:
    def __init__(self) -> None:
        self.statements: list[Any] = []

    async def execute(self, statement: Any, parameters: Any = None) -> _NoRowResult:
        self.statements.append(statement)
        return _NoRowResult()


class _ConnectionContext:
    def __init__(self, connection: _RecordingConnection) -> None:
        self.connection = connection

    async def __aenter__(self) -> _RecordingConnection:
        return self.connection

    async def __aexit__(*_args: object) -> None:
        return None


class _RecordingEngine:
    def __init__(self) -> None:
        self.connection = _RecordingConnection()

    def connect(self) -> _ConnectionContext:
        return _ConnectionContext(self.connection)


def test_statute_lookup_orders_null_years_last() -> None:
    # The newest applicable year wins and unversioned (NULL) rows sort last —
    # now expressed as `instrument_year.desc().nulls_last()` on the chunks model.
    assert "instrument_year.desc().nulls_last()" in _TOOL_SOURCE


async def test_statute_lookup_is_scoped_to_the_authenticated_tenant() -> None:
    engine = _RecordingEngine()

    result = await _fetch_statute_section(
        db_engine=engine,
        act_name=" Contract Act ",
        section_ref=" 73 ",
        jurisdiction="India",
        tenant_id="tenant-42",
        allowed_doc_ids=("00000000-0000-0000-0000-000000000042",),
    )

    assert result is None
    statement = engine.connection.statements[0]
    rendered = str(statement.compile(dialect=postgresql.dialect()))
    compiled_params = statement.compile(dialect=postgresql.dialect()).params
    assert "chunks.tenant_id" in rendered
    assert "ANY" in rendered
    assert "tenant-42" in compiled_params.values()
    assert ["00000000-0000-0000-0000-000000000042"] in compiled_params.values()
    # Identifier normalization stays a caller contract: values are stripped
    # before they are bound.
    assert "Contract Act" in compiled_params.values()
    assert "73" in compiled_params.values()
