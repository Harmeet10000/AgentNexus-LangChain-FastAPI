"""A broken retrieval branch must fail the request; an empty healthy branch must not.

Step 7 of `documents-unified-schema`. Before this change the fused-search path logged a branch
failure and appended an empty rank list, so a request whose keyword branch raised returned `200`
with results silently fused from two modes instead of three. The response was byte-identical to a
request whose keyword branch legitimately matched nothing — which is the reason this is a
correctness fix and not a resilience preference.

Both halves are asserted here, and neither is optional: a failing branch must produce a `Failure`
that names it, and three healthy-but-empty branches must still produce a `Success`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest
from returns.result import Failure, Success

from app.features.documents.dto import UnifiedSearchRequest
from app.features.documents.repository import DocumentRepository

# `_SEARCH_BRANCHES` is private and imported deliberately: the pairing between these labels and
# `asyncio.gather`'s positional results is the thing under test, so asserting on it through a
# public re-export would only test the re-export.
from app.features.documents.service import _SEARCH_BRANCHES, DocumentQueryService  # noqa: PLC2701
from app.shared.result import InfrastructureAppError, ValidationAppError
from app.utils.exceptions import APIException

if TYPE_CHECKING:
    from typing import Any

    from app.shared.result import AppResult

_EMPTY: Success[list[dict[str, Any]]] = Success([])


class _StubRepository:
    """Three branch methods with per-branch outcomes and no database.

    Keyed by branch name rather than by position so that a test which fails `vector_search` is
    stating that fact about the *method*, and the assertion about which label comes back is then
    a real check on the pairing in `_SEARCH_BRANCHES` — not a restatement of it.
    """

    def __init__(
        self,
        *,
        bm25: AppResult[list[dict[str, Any]]],
        vector: AppResult[list[dict[str, Any]]],
        trigram: AppResult[list[dict[str, Any]]],
    ) -> None:
        self._outcomes: dict[str, AppResult[list[dict[str, Any]]]] = {
            "bm25": bm25,
            "vector": vector,
            "trigram": trigram,
        }
        self.called: list[str] = []

    async def bm25_search(self, **_kwargs: object) -> AppResult[list[dict[str, Any]]]:
        self.called.append("bm25")
        return self._outcomes["bm25"]

    async def vector_search(self, **_kwargs: object) -> AppResult[list[dict[str, Any]]]:
        self.called.append("vector")
        return self._outcomes["vector"]

    async def trigram_search(self, **_kwargs: object) -> AppResult[list[dict[str, Any]]]:
        self.called.append("trigram")
        return self._outcomes["trigram"]


def _service(repo: _StubRepository) -> DocumentQueryService:
    # `llm`, `redis` and `graphiti` are unreachable from the fused-search path. Passing `None`
    # through a cast rather than a mock is deliberate: if a future edit makes this path touch the
    # model or the cache, these tests fail with an `AttributeError` on `None` instead of quietly
    # exercising a mock that answers everything.
    return DocumentQueryService(cast("Any", repo), cast("Any", None), None, None)


async def _fuse(repo: _StubRepository) -> AppResult[list[Any]]:
    return await _service(repo)._fuse_search_branches(
        user_id="11111111-1111-1111-1111-111111111111",
        payload=UnifiedSearchRequest(query="indemnity cap"),
        query_embedding=[0.0, 0.1, 0.2],
        filter_params={},
    )


async def test_three_healthy_but_empty_branches_fuse_to_an_empty_success() -> None:
    """An empty result from a healthy branch is not a failure."""
    repo = _StubRepository(bm25=_EMPTY, vector=_EMPTY, trigram=_EMPTY)

    result = await _fuse(repo)

    assert isinstance(result, Success)
    assert result.unwrap() == []
    assert sorted(repo.called) == ["bm25", "trigram", "vector"]


async def test_a_failed_branch_fails_the_whole_retrieval() -> None:
    repo = _StubRepository(
        bm25=Failure(InfrastructureAppError(message="bm25 index is not queryable")),
        vector=_EMPTY,
        trigram=_EMPTY,
    )

    result = await _fuse(repo)

    assert isinstance(result, Failure)


@pytest.mark.parametrize("broken", ["bm25", "vector", "trigram"])
async def test_the_failure_names_the_branch_that_broke(broken: str) -> None:
    """The label must track the method that failed, not the position in a tuple.

    `_fuse_search_branches` pairs `_SEARCH_BRANCHES` with `asyncio.gather`'s results positionally.
    `zip(strict=True)` catches a *length* mismatch and is blind to a *reordering* — swap two names
    and every failure is attributed to the wrong branch with no error anywhere. This
    parametrisation is the check that closes that gap: each branch is broken in turn through its
    own method, so a reordered tuple fails two of the three cases.
    """
    outcomes: dict[str, Any] = dict.fromkeys(_SEARCH_BRANCHES, _EMPTY)
    outcomes[broken] = Failure(InfrastructureAppError(message="branch is down"))
    repo = _StubRepository(**outcomes)

    result = await _fuse(repo)

    assert isinstance(result, Failure)
    error = result.failure()
    assert error.details is not None
    assert error.details["branch"] == broken
    assert error.message.startswith(f"{broken} retrieval branch failed:")


async def test_the_branch_error_keeps_its_own_kind_and_code() -> None:
    """Attribution must not flatten the taxonomy.

    Re-wrapping a branch failure in a fresh `InfrastructureAppError` would turn a validation
    failure in one branch into a retryable 503 for the whole request. The service attaches the
    branch with `model_copy`, so the branch's own kind, code and `retryable` flag survive to
    `app_error_to_exception` and the caller gets the status the branch actually earned.
    """
    repo = _StubRepository(
        bm25=_EMPTY,
        vector=Failure(ValidationAppError(message="embedding width mismatch")),
        trigram=_EMPTY,
    )

    result = await _fuse(repo)

    assert isinstance(result, Failure)
    error = result.failure()
    assert isinstance(error, ValidationAppError)
    assert error.kind == "validation"
    assert error.retryable is False
    assert "embedding width mismatch" in error.message


async def test_the_request_itself_fails_rather_than_answering_200(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Result is an internal seam; the requirement is about the response.

    Asserted through `search` rather than the helper because "fail the request" is the behaviour
    the spec names, and a `Failure` that some future caller logs and swallows would satisfy every
    test above while restoring the exact bug this step removed.
    """

    async def _fake_embed(*_args: object, **_kwargs: object) -> list[float]:
        return [0.0, 0.1, 0.2]

    monkeypatch.setattr("app.features.documents.service.embed_text", _fake_embed)
    repo = _StubRepository(
        bm25=_EMPTY,
        vector=_EMPTY,
        trigram=Failure(InfrastructureAppError(message="trigram operator class is missing")),
    )

    with pytest.raises(APIException) as caught:
        await _service(repo).search(
            user_id="11111111-1111-1111-1111-111111111111",
            payload=UnifiedSearchRequest(query="indemnity cap"),
        )

    assert "trigram" in str(caught.value)


def test_the_branch_labels_match_the_repository_method_names() -> None:
    """Cheap tripwire: the labels are not free-form strings.

    Each label must be the prefix of a real repository search method, so a typo or a renamed
    method surfaces here rather than in a log line nobody reads.
    """
    assert len(_SEARCH_BRANCHES) == 3
    for branch in _SEARCH_BRANCHES:
        assert hasattr(DocumentRepository, f"{branch}_search"), branch
