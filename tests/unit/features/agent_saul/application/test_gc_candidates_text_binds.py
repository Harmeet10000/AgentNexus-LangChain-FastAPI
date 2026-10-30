"""The GC include predicate must compare ->> results as plain TEXT (§8).

``chat_sessions.extra_metadata`` is a ``json`` column. A bare
``extra_metadata.op("->>")(key)`` inherits the JSON type on its result, so the
``pending == 'false'`` / ``== 'true'`` comparisons would bind through the JSON
serializer (rendering ``'"false"'``) and asyncpg would emit a ``$n::JSON``
cast — ``text = json``: an "operator does not exist" error at best, a
never-matching comparison at worst. The repository therefore types the
``->>`` results as ``Text`` (``op(..., return_type=Text)``); this test pins
that contract by compiling the actual statement against the asyncpg dialect.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from returns.result import Success
from sqlalchemy.dialects.postgresql import asyncpg as pg_asyncpg

from app.features.agent_saul.application.transcript.repository import (
    ChatTranscriptRepository,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType

    from sqlalchemy import Select

    from app.features.agent_saul.adapters.persistence.models import ChatSession

_REPO_SOURCE = Path("src/app/features/agent_saul/application/transcript/repository.py").read_text(
    encoding="utf-8"
)


class _ScalarRows:
    def scalars(self) -> _ScalarRows:
        return self

    def all(self) -> Sequence[ChatSession]:
        return []


class _CapturingSession:
    def __init__(self) -> None:
        self.statements: list[Select[tuple[ChatSession]]] = []

    async def __aenter__(self) -> _CapturingSession:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    async def execute(self, statement: Select[tuple[ChatSession]]) -> _ScalarRows:
        self.statements.append(statement)
        return _ScalarRows()


class _SessionFactory:
    def __init__(self, session: _CapturingSession) -> None:
        self._session = session

    def __call__(self) -> _CapturingSession:
        return self._session


def test_the_arrow_right_operator_results_are_explicitly_text_typed() -> None:
    # A revert to a bare op("->>") reintroduces the JSON-typed result and the
    # text = json comparison hazard the compile assertions below pin.
    assert _REPO_SOURCE.count('op("->>", return_type=Text)') == 2


async def test_gc_predicate_binds_plain_text_not_json() -> None:
    session = _CapturingSession()
    repository = ChatTranscriptRepository(_SessionFactory(session))  # type: ignore[arg-type]

    result = await repository.list_gc_candidates(
        older_than=datetime(2026, 1, 1, tzinfo=UTC),
        limit=20,
    )

    assert isinstance(result, Success)
    statement = session.statements[0]
    compiled = statement.compile(dialect=pg_asyncpg.dialect())
    rendered = compiled.string

    # No JSON cast anywhere: the ->> keys and the 'false'/'true' equality
    # literals are all plain text binds under asyncpg.
    assert "::JSON" not in rendered
    # The two pending-flag equality comparisons bind as plain VARCHAR...
    assert len(re.findall(r"= \$\d+::VARCHAR", rendered)) >= 2
    # ...with the literal SQL text 'false'/'true', never the JSON-serialized
    # '"false"' / '"true"' the JSON bind processor would produce.
    param_values = {str(value) for value in compiled.params.values()}
    assert {"false", "true"} <= param_values
    assert '"false"' not in param_values
    assert '"true"' not in param_values
