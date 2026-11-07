"""Unit tests for the retrieval-shaped Graphiti read (graph-fusion-branch).

The fake Graphiti emulates the provider's group scoping: a result is returned
only when its ``group_id`` is among the requested ``group_ids``. The read under
test must rank tenant-scoped chunk identifiers in search order, contribute
nothing when no scope or no relationships are reachable, and fail loudly (not
degrade) on a transport failure — unlike the evidence-policy filter, which
stays untouched and keeps its own swallow-and-empty contract.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from graphiti_core.errors import GraphitiError

from app.retrieval.graphiti.grouping import build_graphiti_group_id
from app.retrieval.graphiti.retrieval_branch import fetch_ranked_chunk_ids

if TYPE_CHECKING:
    from typing import Any

_UUID_A = "00000000-0000-0000-0000-00000000000a"
_UUID_B = "00000000-0000-0000-0000-00000000000b"
_UUID_C = "00000000-0000-0000-0000-00000000000c"


class _FakeGraphiti:
    """Search results filtered by the requested group partitions."""

    def __init__(self, *, episodes_by_group: dict[str, list[str]], error: Exception | None = None):
        self._episodes_by_group = episodes_by_group
        self._error = error
        self.requested_group_ids: list[str] | None = None

    async def search(self, *, query: str, group_ids: list[str], num_results: int) -> list[Any]:
        self.requested_group_ids = list(group_ids)
        if self._error is not None:
            raise self._error
        seen: list[str] = []
        for group_id in group_ids:
            for chunk_id in self._episodes_by_group.get(group_id, []):
                if chunk_id not in seen:
                    seen.append(chunk_id)
        return [
            SimpleNamespace(
                content=f"episode mentioning {chunk_id}",
                source_description=f'{{"chunk_id": "{chunk_id}"}} {chunk_id}',
            )
            for chunk_id in seen[:num_results]
        ]


def _scope_kwargs() -> dict[str, Any]:
    return {
        "allowed_doc_ids": ("doc-a",),
        "tenant_id": "tenant-1",
        "document_versions": [("doc-a", 1, None)],
        "document_contracts": [("doc-a", 1, (None,), True)],
    }


async def test_returns_ranked_chunk_ids_in_search_order() -> None:
    group = build_graphiti_group_id(tenant_id="tenant-1", document_id="doc-a", document_version=1)
    fake = _FakeGraphiti(episodes_by_group={group: [_UUID_B, _UUID_A, _UUID_B, _UUID_C]})

    ranked = await fetch_ranked_chunk_ids(fake, "termination notice", **_scope_kwargs())

    assert fake.requested_group_ids == [group]
    assert ranked == [_UUID_B, _UUID_A, _UUID_C]


async def test_contributes_nothing_without_tenant_scope_or_client() -> None:
    group = build_graphiti_group_id(tenant_id="tenant-1", document_id="doc-a", document_version=1)
    fake = _FakeGraphiti(episodes_by_group={group: [_UUID_A]})

    assert await fetch_ranked_chunk_ids(None, "q", **_scope_kwargs()) == []
    assert await fetch_ranked_chunk_ids(fake, "q", allowed_doc_ids=(), tenant_id="tenant-1") == []
    no_tenant = _scope_kwargs() | {"tenant_id": None}
    assert await fetch_ranked_chunk_ids(fake, "q", **no_tenant) == []


async def test_contributes_nothing_when_no_group_resolves() -> None:
    # A scope without authoritative versions resolves to no partitions.
    fake = _FakeGraphiti(episodes_by_group={})
    ranked = await fetch_ranked_chunk_ids(
        fake,
        "q",
        allowed_doc_ids=("doc-a",),
        tenant_id="tenant-1",
        document_versions=[],
        document_contracts=[],
    )
    assert ranked == []


async def test_contributes_nothing_when_no_relationships_reachable() -> None:
    fake = _FakeGraphiti(episodes_by_group={})
    ranked = await fetch_ranked_chunk_ids(fake, "unrelated query", **_scope_kwargs())
    assert ranked == []


async def test_transport_failure_raises_instead_of_degrading() -> None:
    fake = _FakeGraphiti(episodes_by_group={}, error=GraphitiError("neo4j down"))
    with pytest.raises(GraphitiError):
        await fetch_ranked_chunk_ids(fake, "q", **_scope_kwargs())
