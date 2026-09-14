"""Phrase post-filter lives in the keyword leg (retrieval-sql task 6.1).

The keyword extension has no phrase syntax, so the leg over-fetches on
relevance and post-filters with an escaped literal `LIKE` pattern. A phrase
containing a wildcard metacharacter must match literally, and a chunk holding
the words separately but not the phrase must be excluded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from returns.result import Success

from app.features.documents.constants import PHRASE_OVERFETCH_MULTIPLE
from app.features.documents.repository import _phrase_like_pattern

if TYPE_CHECKING:
    from typing import Any

    from app.features.documents.repository import DocumentRepository


def test_phrase_pattern_escapes_wildcard_and_escape_characters() -> None:
    assert _phrase_like_pattern("100% guarantee") == "%100\\% guarantee%"
    assert _phrase_like_pattern("clause_2") == "%clause\\_2%"
    assert _phrase_like_pattern("back\\slash") == "%back\\\\slash%"


def test_phrase_pattern_wraps_in_contains() -> None:
    assert _phrase_like_pattern("indemnity cap") == "%indemnity cap%"


class _CapturingSession:
    """Fake session: records the statement and params, returns canned rows."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.statements: list[str] = []
        self.params: list[dict[str, Any]] = []

    async def execute(
        self, statement: Any, params: dict[str, Any] | None = None
    ) -> Any:
        from sqlalchemy.sql.elements import TextClause

        assert isinstance(statement, TextClause)
        self.statements.append(str(statement))
        self.params.append(dict(params or {}))

        class _Result:
            def __init__(self, rows: list[dict[str, Any]]) -> None:
                self._rows = rows

            def mappings(self) -> _Result:
                return self

            def all(self) -> list[dict[str, Any]]:
                return list(self._rows)

        return _Result(self._rows)


def _repository(session: _CapturingSession) -> DocumentRepository:
    from app.features.documents.repository import DocumentRepository

    return DocumentRepository(session)  # type: ignore[arg-type]


async def test_keyword_leg_filters_phrase_with_escaped_pattern() -> None:
    rows = [{"chunk_id": f"c-{i}", "score": -float(i)} for i in range(8)]
    session = _CapturingSession(rows)
    repo = _repository(session)

    result = await repo.bm25_search(
        user_id="user-1",
        query="guarantee",
        candidate_limit=2,
        filter_params={
            "document_ids": [],
            "chunk_ids": [],
            "document_kind": None,
            "jurisdiction": None,
            "contract_type": None,
            "clause_type": None,
            "require_graphiti_verified": False,
            "metadata_filter": "{}",
            "parties_filter": "[]",
        },
        exact_phrase="100% guarantee",
    )

    assert isinstance(result, Success)
    assert len(session.statements) == 1
    sql = session.statements[0]
    assert "LIKE CAST(:phrase_pattern AS text) ESCAPE" in sql
    params = session.params[0]
    assert params["phrase_pattern"] == "%100\\% guarantee%"
    # Over-fetch: the leg reads a multiple of the candidate budget, then the
    # method truncates back to it after the LIKE post-filter.
    assert params["fetch_limit"] == 2 * PHRASE_OVERFETCH_MULTIPLE
    assert [row["chunk_id"] for row in result.unwrap()] == ["c-0", "c-1"]


async def test_keyword_leg_without_phrase_keeps_single_limit() -> None:
    session = _CapturingSession([{"chunk_id": "c-0", "score": -1.0}])
    repo = _repository(session)

    result = await repo.bm25_search(
        user_id="user-1",
        query="guarantee",
        candidate_limit=2,
        filter_params={
            "document_ids": [],
            "chunk_ids": [],
            "document_kind": None,
            "jurisdiction": None,
            "contract_type": None,
            "clause_type": None,
            "require_graphiti_verified": False,
            "metadata_filter": "{}",
            "parties_filter": "[]",
        },
    )

    assert isinstance(result, Success)
    assert session.params[0]["phrase_pattern"] is None
    assert session.params[0]["fetch_limit"] == 2
