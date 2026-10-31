"""Unit tests for the Graphiti relationship-edge read surface.

The fake Graphiti here emulates the provider's own group scoping: an episode
is only returned when its ``group_id`` is among the requested ``group_ids``.
That makes the cross-tenant tests real — a tenant's fetch can only ever see
episodes whose group partition it asked for, and the tests pin that the
service asks only for partitions derived from the authorized scope.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from graphiti_core.errors import GraphitiError

from app.retrieval.graphiti.client import fetch_relationship_edges
from app.retrieval.graphiti.grouping import build_graphiti_group_id

if TYPE_CHECKING:
    from typing import Any


def _edge_body(
    from_entity: str,
    relationship: str,
    to_entity: str,
    *,
    char_start: int | None = 1042,
    char_end: int | None = 1110,
) -> str:
    body = f"{from_entity} {relationship} {to_entity}. Source clause: 2.1. "
    body += "Citation: clause 2.1. Confidence: 0.90."
    if char_start is not None and char_end is not None:
        body += f" Grounding: chars {char_start}-{char_end}."
    return body


def _edge_episode(
    edge_id: str,
    *,
    group_id: str,
    doc_id: str = "doc-a",
    body: str | None = None,
    source_type: str = "relationship_edge",
    clause_id: str = "2.1",
) -> SimpleNamespace:
    return SimpleNamespace(
        uuid=edge_id,
        name=f"edge:{edge_id}",
        content=body or _edge_body("Acme Corp", "obliges", "GlobalTech Ltd"),
        source_description=json.dumps(
            {
                "type": source_type,
                "doc_id": doc_id,
                "clause_id": clause_id,
                "confidence": 0.9,
            }
        ),
        group_id=group_id,
        score=0.9,
        created_at=None,
    )


class _ScopingGraphiti:
    """Fake client that answers exactly like the provider's group partition."""

    def __init__(self, episodes: list[SimpleNamespace]) -> None:
        self._episodes = episodes
        self.calls: list[dict[str, Any]] = []

    async def search(
        self, *, query: str, group_ids: list[str], num_results: int
    ) -> list[SimpleNamespace]:
        self.calls.append(
            {"query": query, "group_ids": list(group_ids), "num_results": num_results}
        )
        return [episode for episode in self._episodes if episode.group_id in group_ids][
            :num_results
        ]


class _ExplodingGraphiti:
    async def search(self, **_kwargs: object) -> list[SimpleNamespace]:
        message = "graph is down"
        raise GraphitiError(message)


_TENANT_A_GROUP = build_graphiti_group_id(
    tenant_id="tenant-a", document_id="doc-a", document_version=1, contract_index=0
)
_TENANT_B_GROUP = build_graphiti_group_id(
    tenant_id="tenant-b", document_id="doc-b", document_version=1, contract_index=0
)


async def test_empty_allowlist_fails_closed_without_a_search_call() -> None:
    graphiti = _ScopingGraphiti([_edge_episode("edge-1", group_id=_TENANT_A_GROUP)])

    result = await fetch_relationship_edges(graphiti, "payment", allowed_doc_ids=())

    assert result == ()
    assert graphiti.calls == []


async def test_allowlist_without_tenant_never_issues_an_unscoped_search() -> None:
    graphiti = _ScopingGraphiti([_edge_episode("edge-1", group_id=_TENANT_A_GROUP)])

    result = await fetch_relationship_edges(
        graphiti,
        "payment",
        allowed_doc_ids=("doc-a",),
        tenant_id=None,
        document_versions=(("doc-a", 1),),
    )

    assert result == ()
    assert graphiti.calls == []


async def test_scope_without_resolvable_groups_reads_nothing() -> None:
    # A document with no authorized version resolves to no partitions; the
    # read must fail closed rather than widen to a legacy group.
    graphiti = _ScopingGraphiti([_edge_episode("edge-1", group_id=_TENANT_A_GROUP)])

    result = await fetch_relationship_edges(
        graphiti,
        "payment",
        allowed_doc_ids=("doc-a",),
        tenant_id="tenant-a",
    )

    assert result == ()
    assert graphiti.calls == []


async def test_edge_episode_parses_into_a_typed_record() -> None:
    graphiti = _ScopingGraphiti([_edge_episode("edge-1", group_id=_TENANT_A_GROUP)])

    result = await fetch_relationship_edges(
        graphiti,
        "payment",
        allowed_doc_ids=("doc-a",),
        tenant_id="tenant-a",
        document_versions=(("doc-a", 1),),
        document_contracts=(("doc-a", 1, (0,), False),),
    )

    assert len(result) == 1
    edge = result[0]
    assert edge.edge_id == "edge-1"
    assert edge.document_id == "doc-a"
    assert edge.from_entity == "Acme Corp"
    assert edge.relationship == "obliges"
    assert edge.to_entity == "GlobalTech Ltd"
    assert edge.clause_id == "2.1"
    assert edge.char_start == 1042
    assert edge.char_end == 1110
    assert edge.confidence == pytest.approx(0.9)
    assert edge.group_id == _TENANT_A_GROUP


async def test_non_edge_episodes_and_untypeable_bodies_are_dropped() -> None:
    clause_episode = _edge_episode(
        "clause-1",
        group_id=_TENANT_A_GROUP,
        body="The Employer shall pay within fourteen days.",
        source_type="clause_episode",
    )
    no_relationship = _edge_episode(
        "edge-bad",
        group_id=_TENANT_A_GROUP,
        body="Something something nothing to type.",
    )
    missing_source = SimpleNamespace(
        uuid="edge-nosrc",
        name="edge:x",
        content=_edge_body("Acme Corp", "obliges", "GlobalTech Ltd"),
        source_description="not json",
        group_id=_TENANT_A_GROUP,
        score=0.5,
        created_at=None,
    )
    graphiti = _ScopingGraphiti([clause_episode, no_relationship, missing_source])

    result = await fetch_relationship_edges(
        graphiti,
        "payment",
        allowed_doc_ids=("doc-a",),
        tenant_id="tenant-a",
        document_versions=(("doc-a", 1),),
        document_contracts=(("doc-a", 1, (0,), False),),
    )

    assert result == ()


async def test_transport_failure_degrades_to_an_empty_tuple() -> None:
    result = await fetch_relationship_edges(
        _ExplodingGraphiti(),
        "payment",
        allowed_doc_ids=("doc-a",),
        tenant_id="tenant-a",
        document_versions=(("doc-a", 1),),
        document_contracts=(("doc-a", 1, (0,), False),),
    )

    assert result == ()


async def test_a_tenant_cannot_read_another_tenant_edges() -> None:
    # Tenant B's edges exist in the store, but the read is scoped to tenant A's
    # authorized partitions: the fake provider never returns them, and the
    # service never even asks for tenant B's group id.
    graphiti = _ScopingGraphiti(
        [
            _edge_episode("edge-a", group_id=_TENANT_A_GROUP, doc_id="doc-a"),
            _edge_episode(
                "edge-b",
                group_id=_TENANT_B_GROUP,
                doc_id="doc-b",
                body=_edge_body("BetaCorp", "obliges", "Gamma LLC"),
            ),
        ]
    )

    result = await fetch_relationship_edges(
        graphiti,
        "payment",
        allowed_doc_ids=("doc-a",),
        tenant_id="tenant-a",
        document_versions=(("doc-a", 1),),
        document_contracts=(("doc-a", 1, (0,), False),),
    )

    assert [edge.edge_id for edge in result] == ["edge-a"]
    for call in graphiti.calls:
        assert _TENANT_B_GROUP not in call["group_ids"]
        assert all(group_id.startswith("tenant_") for group_id in call["group_ids"])


@pytest.mark.parametrize(
    ("relationship", "expected"),
    [("OBLIGES", "obliges"), ("Indemnifies", "indemnifies")],
)
async def test_relationship_word_is_matched_case_insensitively(
    relationship: str, expected: str
) -> None:
    graphiti = _ScopingGraphiti(
        [
            _edge_episode(
                "edge-1",
                group_id=_TENANT_A_GROUP,
                body=_edge_body("Acme Corp", relationship, "GlobalTech Ltd"),
            )
        ]
    )

    result = await fetch_relationship_edges(
        graphiti,
        "payment",
        allowed_doc_ids=("doc-a",),
        tenant_id="tenant-a",
        document_versions=(("doc-a", 1),),
        document_contracts=(("doc-a", 1, (0,), False),),
    )

    assert result[0].relationship == expected
