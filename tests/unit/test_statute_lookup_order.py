"""The statute point lookup prefers dated rows over unversioned rows."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from app.shared.langchain_layer.agents.tools.retrieve_statute_section import (
    _fetch_statute_section,
)

if TYPE_CHECKING:
    from typing import Any


class _NoRowResult:
    def fetchone(self) -> None:
        return None


class _RecordingConnection:
    def __init__(self) -> None:
        self.query = ""
        self.parameters: dict[str, object] = {}

    async def execute(self, query: Any, parameters: dict[str, object]) -> _NoRowResult:
        self.query = str(query)
        self.parameters = parameters
        return _NoRowResult()


class _ConnectionContext:
    def __init__(self, connection: _RecordingConnection) -> None:
        self.connection = connection

    async def __aenter__(self) -> _RecordingConnection:
        return self.connection

    async def __aexit__(self, *_args: object) -> None:
        return None


class _RecordingEngine:
    def __init__(self) -> None:
        self.connection = _RecordingConnection()

    def connect(self) -> _ConnectionContext:
        return _ConnectionContext(self.connection)


def test_statute_lookup_orders_null_years_last() -> None:
    source = Path(
        "src/app/shared/langchain_layer/agents/tools/retrieve_statute_section.py"
    ).read_text(encoding="utf-8")
    assert "ORDER BY instrument_year DESC NULLS LAST" in source


async def test_statute_lookup_is_scoped_to_the_authenticated_tenant() -> None:
    engine = _RecordingEngine()

    result = await _fetch_statute_section(
        db_engine=engine,
        act_name=" Contract Act ",
        section_ref=" 73 ",
        jurisdiction="India",
        user_id="tenant-42",
    )

    assert result is None
    assert "user_id = :user_id" in engine.connection.query
    assert engine.connection.parameters == {
        "user_id": "tenant-42",
        "act_name": "Contract Act",
        "section_ref": "73",
    }
