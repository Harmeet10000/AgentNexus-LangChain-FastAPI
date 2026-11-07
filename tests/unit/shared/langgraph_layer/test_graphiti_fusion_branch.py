"""Branch-registry registration and enabled-path behavior for the graph branch.

Pins the `graph-fusion-branch` spec at the registry contract: the branch is a
registration (gather/zip/fusion untouched), default deployments run the three
mandatory modes only, disabled is not a provisioning failure, an enabled branch
failure fails the request naming the branch, and the branch's contribution is
recorded in execution diagnostics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from graphiti_core.errors import GraphitiError
from returns.result import Success

from app.config.settings import get_settings
from app.features.documents.application.queries.documents import (
    GRAPHITI_BRANCH_NAME,
    RETRIEVAL_BRANCHES,
    fused_branches,
    retrieve_fused,
)

if TYPE_CHECKING:
    from typing import Any

_UUID = "00000000-0000-0000-0000-00000000000a"


class _RecordingRepo:
    """Three-leg stub; records which branch methods ran."""

    def __init__(self) -> None:
        self.called: list[str] = []

    async def bm25_search(self, **_kwargs: Any) -> Any:
        self.called.append("bm25")
        return Success([{"chunk_id": "a", "score": -5.0}])

    async def vector_search(self, **_kwargs: Any) -> Any:
        self.called.append("vector")
        return Success([{"chunk_id": "b", "score": 0.9}])

    async def trigram_search(self, **_kwargs: Any) -> Any:
        self.called.append("trigram")
        return Success([{"chunk_id": "c", "score": 0.7}])

    async def fetch_chunks_by_ids(self, chunk_ids: Any, **_kwargs: Any) -> Any:
        return Success({cid: {"chunk_id": cid} for cid in chunk_ids})


class _FakeGraphiti:
    def __init__(self, chunk_ids: list[str] | None = None, error: Exception | None = None) -> None:
        self._chunk_ids = chunk_ids or []
        self._error = error

    async def search(self, *, query: str, group_ids: list[str], num_results: int) -> list[Any]:
        if self._error is not None:
            raise self._error
        from types import SimpleNamespace

        return [SimpleNamespace(content=cid, source_description=cid) for cid in self._chunk_ids]


def test_graph_branch_is_a_registration_on_the_shared_tuple() -> None:
    assert [branch.name for branch in RETRIEVAL_BRANCHES] == ["bm25", "vector", "trigram"]
    enabled = fused_branches(object())
    assert [branch.name for branch in enabled] == [
        "bm25",
        "vector",
        "trigram",
        GRAPHITI_BRANCH_NAME,
    ]
    # The mandatory legs are the same records — no parallel registry exists.
    assert enabled[:3] == RETRIEVAL_BRANCHES


def test_flag_defaults_off_and_disabled_is_not_a_provisioning_failure() -> None:
    settings = get_settings()
    assert settings.DOCUMENTS_GRAPHITI_FUSION_ENABLED is False
    # The disabled branch set is exactly the mandatory modes; nothing surfaces
    # through a missing-capability error path.
    assert fused_branches(None) is RETRIEVAL_BRANCHES


async def test_default_off_runs_three_mandatory_branches_only() -> None:
    repo = _RecordingRepo()
    result = await retrieve_fused(
        repo=repo,
        user_id="user-1",
        query_text="termination",
        query_embedding=[0.1],
        candidate_limit=10,
        limit=5,
        filter_params={},
    )
    assert isinstance(result, Success)
    assert repo.called == ["bm25", "vector", "trigram"]


async def test_enabled_branch_contributes_and_diagnostics_record_it() -> None:
    repo = _RecordingRepo()
    diagnostics: dict[str, Any] = {}
    result = await retrieve_fused(
        repo=repo,
        user_id="user-1",
        query_text="termination",
        query_embedding=[0.1],
        candidate_limit=10,
        limit=5,
        filter_params={"document_ids": ["doc-1"]},
        graphiti=_FakeGraphiti(chunk_ids=[_UUID]),
        document_versions=[("doc-1", 1, None)],
        document_contracts=[("doc-1", 1, (None,), True)],
        branch_diagnostics=diagnostics,
    )
    assert isinstance(result, Success)
    assert repo.called == ["bm25", "vector", "trigram"]
    by_branch = {item["branch"]: item for item in diagnostics["branches_executed"]}
    assert [branch["branch"] for branch in diagnostics["branches_executed"]] == [
        "bm25",
        "vector",
        "trigram",
        GRAPHITI_BRANCH_NAME,
    ]
    assert by_branch[GRAPHITI_BRANCH_NAME]["executed"] is True
    assert by_branch[GRAPHITI_BRANCH_NAME]["identifiers_contributed"] == 1


async def test_enabled_branch_failure_fails_request_naming_the_branch() -> None:
    repo = _RecordingRepo()
    result = await retrieve_fused(
        repo=repo,
        user_id="user-1",
        query_text="termination",
        query_embedding=[0.1],
        candidate_limit=10,
        limit=5,
        filter_params={"document_ids": ["doc-1"]},
        graphiti=_FakeGraphiti(error=GraphitiError("graph unavailable")),
        document_versions=[("doc-1", 1, None)],
        document_contracts=[("doc-1", 1, (None,), True)],
    )
    assert not isinstance(result, Success)
    failure = result.failure()
    assert GRAPHITI_BRANCH_NAME in failure.message
    assert failure.details.get("branch") == GRAPHITI_BRANCH_NAME


async def test_enabled_but_empty_branch_does_not_dilute_ranking() -> None:
    """Spec: empty traversal does not dilute ranking, and participation is honest.

    An enabled branch that matches nothing contributes an empty rank list:
    the fused ranking equals the three-branch ranking for the same query, and
    the branch diagnostics record the graph branch as executed-but-empty
    (participating-but-empty), distinct from the branches that matched.
    """
    base_kwargs: dict[str, Any] = {
        "user_id": "user-1",
        "query_text": "termination",
        "query_embedding": [0.1],
        "candidate_limit": 10,
        "limit": 5,
        "filter_params": {"document_ids": ["doc-1"]},
        "document_versions": [("doc-1", 1, None)],
        "document_contracts": [("doc-1", 1, (None,), True)],
    }
    empty_diagnostics: dict[str, Any] = {}
    empty_result = await retrieve_fused(
        repo=_RecordingRepo(),
        graphiti=_FakeGraphiti(chunk_ids=[]),
        branch_diagnostics=empty_diagnostics,
        weights=[1.0, 1.0, 1.2, 0.1],
        **base_kwargs,
    )
    graphless_result = await retrieve_fused(
        repo=_RecordingRepo(),
        weights=[1.0, 1.0, 1.2],
        **base_kwargs,
    )

    assert isinstance(empty_result, Success)
    assert isinstance(graphless_result, Success)
    empty_ids = [item.chunk_id for item in empty_result.unwrap()[0]]
    graphless_ids = [item.chunk_id for item in graphless_result.unwrap()[0]]
    # An empty fourth leg contributes no rank entries, so the fused ranking is
    # exactly the three-branch ranking for this query.
    assert empty_ids == graphless_ids
    by_branch = {item["branch"]: item for item in empty_diagnostics["branches_executed"]}
    assert by_branch[GRAPHITI_BRANCH_NAME] == {
        "branch": GRAPHITI_BRANCH_NAME,
        "executed": True,
        "identifiers_contributed": 0,
    }


def test_graphiti_weight_cites_the_recorded_baseline() -> None:
    """The fusion weight is the measured constant, not an asserted one.

    ``RRF_WEIGHT_GRAPHITI`` must equal the ``measured_weight`` recorded in the
    deterministic weight-sweep report (argmax of the sweep on recall@10, MRR,
    nDCG@10 over the grown golden corpus) — a changed weight fails here until
    the report is regenerated with the new measurement.
    """
    import json
    from pathlib import Path

    from app.features.documents.domain.constants import RRF_WEIGHT_GRAPHITI

    report_path = Path("evals/reports/graphiti_branch_legal_retrieval_v2_wave4.measurements.json")
    measured = json.loads(report_path.read_text(encoding="utf-8"))
    assert measured["measured_weight"] == RRF_WEIGHT_GRAPHITI
    assert RRF_WEIGHT_GRAPHITI in {point["w_graph"] for point in measured["weight_sweep"]["points"]}
    assert "RRF_WEIGHT_GRAPHITI" in measured["decision"]


async def test_four_leg_fusion_consumes_four_weights_without_a_second_implementation() -> None:
    repo = _RecordingRepo()
    result = await retrieve_fused(
        repo=repo,
        user_id="user-1",
        query_text="termination",
        query_embedding=[0.1],
        candidate_limit=10,
        limit=5,
        filter_params={"document_ids": ["doc-1"]},
        weights=[1.0, 1.0, 1.2, 0.1],
        graphiti=_FakeGraphiti(chunk_ids=[_UUID]),
        document_versions=[("doc-1", 1, None)],
        document_contracts=[("doc-1", 1, (None,), True)],
    )
    assert isinstance(result, Success)
    fused, _ = result.unwrap()
    # c: trigram 1.2 at rank 1; a and b: one unweighted rank-1 leg each (tie broken
    # by id); the UUID: only the 0.1 graph weight. The graph leg contributed over
    # the same identifier space and its weight demonstrably applied.
    assert [item.chunk_id for item in fused] == ["c", "a", "b", _UUID]
