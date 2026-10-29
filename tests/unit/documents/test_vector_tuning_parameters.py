"""Pin vector-leg query-time tuning on the fused path (retrieval-sql task 4.2).

The structural half: every vector-leg execution sets both approximate-search
parameters in its own transaction before the scan they govern. The recall-difference
half of 4.2 is recorded as an accepted deviation in the change review — on the
provisioned instance both GUCs are placeholders (`SET` accepts even `'nonsense'`,
`pg_settings` lists no `diskann.*` parameter), so no recall difference can exist
there. These tests keep the statements present (older vectorscale honours them)
and ordered (tuning before scan).
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

from returns.result import Success

from app.features.documents.adapters.persistence.postgres.document_repository import (
    DocumentRepository,
)
from app.features.documents.domain import constants

if TYPE_CHECKING:
    from typing import Any


class _RecordedResult:
    def mappings(self) -> _RecordedResult:
        return self

    def all(self) -> list[dict[str, Any]]:
        return []


class _RecordingSession:
    """Minimal async-session stand-in recording executed statements in order."""

    def __init__(self) -> None:
        self.statements: list[str] = []

    async def execute(self, statement: Any, *args: Any, **kwargs: Any) -> _RecordedResult:
        self.statements.append(str(statement.text if hasattr(statement, "text") else statement))
        return _RecordedResult()


def test_vector_leg_sets_both_tuning_parameters_before_the_scan() -> None:
    source = inspect.getsource(DocumentRepository.vector_search)

    assert "SET LOCAL diskann.query_search_list_size" in source
    assert "SET LOCAL diskann.query_rescore" in source
    assert "DISKANN_QUERY_SEARCH_LIST_SIZE" in source
    assert "DISKANN_QUERY_RESCORE" in source


async def test_vector_leg_tuning_executes_in_order_on_one_session() -> None:
    session = _RecordingSession()
    repository = DocumentRepository(session)  # type: ignore[arg-type]

    result = await repository.vector_search(
        user_id="user-1",
        embedding=[0.0, 0.0],
        candidate_limit=10,
        filter_params={},
    )

    assert isinstance(result, Success)
    assert len(session.statements) == 3
    assert session.statements[0] == (
        f"SET LOCAL diskann.query_search_list_size = {constants.DISKANN_QUERY_SEARCH_LIST_SIZE}"
    )
    assert session.statements[1] == (
        f"SET LOCAL diskann.query_rescore = {constants.DISKANN_QUERY_RESCORE}"
    )
    assert "FROM chunks AS c" in session.statements[2]
