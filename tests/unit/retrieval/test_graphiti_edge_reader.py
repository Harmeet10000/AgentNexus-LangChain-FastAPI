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

from app.retrieval.graphiti import client as client_module
from app.retrieval.graphiti.client import (
    _parse_relationship_edge,
    fetch_relationship_edges,
)
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


class _FakeGraphiti:
    """Fake client exposing only the graph driver the reader needs."""

    def __init__(self) -> None:
        self.driver: object = object()
        self.calls: list[dict[str, Any]] = []


def _install_episode_source(
    monkeypatch: pytest.MonkeyPatch, episodes: list[SimpleNamespace]
) -> _FakeGraphiti:
    """Patch the episode read surface; records the scoped groups it reads."""

    async def fake_read(
        _driver: object, *, group_ids: list[str], limit: int, uuid_cursor: str | None = None
    ) -> list[SimpleNamespace]:
        source.calls.append({"group_ids": list(group_ids), "limit": limit})
        return [episode for episode in episodes if episode.group_id in group_ids][:limit]

    source = _FakeGraphiti()
    monkeypatch.setattr(client_module, "_read_episodes_by_group", fake_read)
    return source


class _ExplodingGraphiti:
    driver: object = object()


def _install_exploding_source(monkeypatch: pytest.MonkeyPatch) -> _ExplodingGraphiti:
    async def explode(
        _driver: object, *, group_ids: list[str], limit: int, uuid_cursor: str | None = None
    ) -> list[SimpleNamespace]:
        message = "graph is down"
        raise GraphitiError(message)

    graphiti = _ExplodingGraphiti()
    monkeypatch.setattr(client_module, "_read_episodes_by_group", explode)
    return graphiti


_TENANT_A_GROUP = build_graphiti_group_id(
    tenant_id="tenant-a", document_id="doc-a", document_version=1, contract_index=0
)
_TENANT_B_GROUP = build_graphiti_group_id(
    tenant_id="tenant-b", document_id="doc-b", document_version=1, contract_index=0
)


async def test_empty_allowlist_fails_closed_without_a_search_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graphiti = _install_episode_source(
        monkeypatch, [_edge_episode("edge-1", group_id=_TENANT_A_GROUP)]
    )

    result = await fetch_relationship_edges(graphiti, "payment", allowed_doc_ids=())

    assert result == ()
    assert graphiti.calls == []


async def test_allowlist_without_tenant_never_issues_an_unscoped_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graphiti = _install_episode_source(
        monkeypatch, [_edge_episode("edge-1", group_id=_TENANT_A_GROUP)]
    )

    result = await fetch_relationship_edges(
        graphiti,
        "payment",
        allowed_doc_ids=("doc-a",),
        tenant_id=None,
        document_versions=(("doc-a", 1),),
    )

    assert result == ()
    assert graphiti.calls == []


async def test_scope_without_resolvable_groups_reads_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A document with no authorized version resolves to no partitions; the
    # read must fail closed rather than widen to a legacy group.
    graphiti = _install_episode_source(
        monkeypatch, [_edge_episode("edge-1", group_id=_TENANT_A_GROUP)]
    )

    result = await fetch_relationship_edges(
        graphiti,
        "payment",
        allowed_doc_ids=("doc-a",),
        tenant_id="tenant-a",
    )

    assert result == ()
    assert graphiti.calls == []


async def test_edge_episode_parses_into_a_typed_record(monkeypatch: pytest.MonkeyPatch) -> None:
    graphiti = _install_episode_source(
        monkeypatch, [_edge_episode("edge-1", group_id=_TENANT_A_GROUP)]
    )

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


async def test_non_edge_episodes_and_untypeable_bodies_are_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    graphiti = _install_episode_source(
        monkeypatch, [clause_episode, no_relationship, missing_source]
    )

    result = await fetch_relationship_edges(
        graphiti,
        "payment",
        allowed_doc_ids=("doc-a",),
        tenant_id="tenant-a",
        document_versions=(("doc-a", 1),),
        document_contracts=(("doc-a", 1, (0,), False),),
    )

    assert result == ()


async def test_transport_failure_degrades_to_an_empty_tuple(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = await fetch_relationship_edges(
        _install_exploding_source(monkeypatch),
        "payment",
        allowed_doc_ids=("doc-a",),
        tenant_id="tenant-a",
        document_versions=(("doc-a", 1),),
        document_contracts=(("doc-a", 1, (0,), False),),
    )

    assert result == ()


async def test_a_tenant_cannot_read_another_tenant_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    # Tenant B's edges exist in the store, but the read is scoped to tenant A's
    # authorized partitions: the fake provider never returns them, and the
    # service never even asks for tenant B's group id.
    graphiti = _install_episode_source(
        monkeypatch,
        [
            _edge_episode("edge-a", group_id=_TENANT_A_GROUP, doc_id="doc-a"),
            _edge_episode(
                "edge-b",
                group_id=_TENANT_B_GROUP,
                doc_id="doc-b",
                body=_edge_body("BetaCorp", "obliges", "Gamma LLC"),
            ),
        ],
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
    relationship: str, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    graphiti = _install_episode_source(
        monkeypatch,
        [
            _edge_episode(
                "edge-1",
                group_id=_TENANT_A_GROUP,
                body=_edge_body("Acme Corp", relationship, "GlobalTech Ltd"),
            )
        ],
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


async def test_structured_triple_survives_abbreviated_entity_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """New rows read the triple from source_description, never prose.

    'Acme Corp. obliges GlobalTech Ltd.' contains sentence punctuation; the
    prose fallback must not be load-bearing for rows that carry the
    structured fields.
    """
    body = _edge_body("Acme Corp.", "obliges", "GlobalTech Ltd.")
    episode = _edge_episode(
        "edge-abbrev",
        group_id="g1",
        body=body,
    )
    meta = json.loads(episode.source_description)
    meta["from_entity"] = "Acme Corp."
    meta["relationship"] = "obliges"
    meta["to_entity"] = "GlobalTech Ltd."
    episode.source_description = json.dumps(meta)
    record = _parse_relationship_edge(episode)
    assert record is not None
    assert record.from_entity == "Acme Corp."
    assert record.relationship == "obliges"
    assert record.to_entity == "GlobalTech Ltd."
    assert record.contract_index is None  # not asserted by legacy fixture


async def test_structured_fields_carry_contract_and_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """contract_index/document_version ride in source_description for scoping."""
    episode = _edge_episode("edge-scoped", group_id="g1")
    meta = json.loads(episode.source_description)
    meta["from_entity"] = "Acme Corp."
    meta["relationship"] = "obliges"
    meta["to_entity"] = "GlobalTech Ltd."
    meta["contract_index"] = 2
    meta["document_version"] = 3
    episode.source_description = json.dumps(meta)
    record = _parse_relationship_edge(episode)
    assert record is not None
    assert record.contract_index == 2
    assert record.document_version == 3


async def test_legacy_prose_rows_still_parse_despite_abbreviations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-structured rows fall back to prose — the vocabulary scan spans the
    whole first line, so 'Acme Corp. obliges GlobalTech Ltd.' parses instead
    of being truncated at the abbreviation period."""
    episode = _edge_episode(
        "edge-legacy",
        group_id="g1",
        body=_edge_body("Acme Corp.", "obliges", "GlobalTech Ltd."),
    )
    record = _parse_relationship_edge(episode)
    assert record is not None
    assert record.relationship == "obliges"
    assert "Corp" in record.from_entity
    assert "GlobalTech" in record.to_entity


async def test_pagination_keeps_collecting_until_the_edge_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chunk/clause episodes share the groups; they must not crowd edges out.

    Page 1 is full of non-edge episodes; page 2 carries the relationship.
    The read must keep paging until the edge cap is met.
    """
    pages = [
        [
            SimpleNamespace(
                uuid=f"clause-{index}",
                name=f"clause:{index}",
                content="The Employer shall pay within fourteen days.",
                source_description=json.dumps({"type": "clause_episode"}),
                group_id=_TENANT_A_GROUP,
                score=0.5,
                created_at=None,
            )
            for index in range(64)
        ],
        [_edge_episode("edge-1", group_id=_TENANT_A_GROUP)],
    ]
    seen_cursors: list[str | None] = []

    async def paged_read(
        _driver: object,
        *,
        group_ids: list[str],
        limit: int,
        uuid_cursor: str | None = None,
    ) -> list[SimpleNamespace]:
        del group_ids, limit
        seen_cursors.append(uuid_cursor)
        return pages.pop(0) if pages else []

    monkeypatch.setattr(client_module, "_read_episodes_by_group", paged_read)
    graphiti = _FakeGraphiti()

    result = await fetch_relationship_edges(
        graphiti,
        "payment",
        allowed_doc_ids=("doc-a",),
        tenant_id="tenant-a",
        document_versions=(("doc-a", 1),),
        document_contracts=(("doc-a", 1, (0,), False),),
    )

    assert [record.edge_id for record in result] == ["edge-1"]
    assert seen_cursors == [None, "clause-63"]
