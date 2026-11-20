"""#285's keyed channel reducers: Send-fan-out replays must converge, not duplicate.

``operator.add`` channels append on every retry/replay with no way to
converge; a Send fan-out that retries re-appends its whole batch. The keyed
merge reducers keep insertion order, update a replayed identity in place
(last-write-wins), and append only genuinely-new identities. ``errors``
stays ``operator.add`` on purpose — it is an audit trail where a replayed
entry is an honest record.
"""

from __future__ import annotations

import pytest

from app.features.agent_saul.domain.state import (
    Citation,
    CitedEntity,
    ClauseSegment,
    ClauseType,
    EntityType,
    LegalRelationship,
    merge_entities,
    merge_relationships,
    merge_segments,
)
from app.retrieval.graphiti.schemas import RelationshipType

_CLAIM = Citation(claim="test claim", source="SRC-01:p1:c1", confidence=0.9)


def _segment(clause_id: str, section: str = "§1") -> ClauseSegment:
    return ClauseSegment(
        clause_id=clause_id,
        clause_type=ClauseType.GOVERNING_LAW,
        section_ref=section,
        start_char=0,
        end_char=10,
        text="runtime only",
    )


def _entity(entity_id: str) -> CitedEntity:
    return CitedEntity(
        entity_id=entity_id,
        entity_type=EntityType.OBLIGATION,
        value="an obligation",
        clause_id="SRC-01:p1:c1",
        citation=_CLAIM,
    )


def _relationship(edge_id: str, confidence: float = 0.9) -> LegalRelationship:
    return LegalRelationship(
        edge_id=edge_id,
        from_node="e-1",
        relationship=RelationshipType.OBLIGES,
        to_node="e-2",
        clause_id="SRC-01:p1:c1",
        citation=Citation(claim="test claim", source="SRC-01:p1:c1", confidence=confidence),
        contract_index=None,
    )


def test_segment_replay_updates_in_place_instead_of_appending() -> None:
    existing = [_segment("c-1"), _segment("c-2")]
    replayed = [_segment("c-1", section="§9")]

    merged = merge_segments(existing, replayed)

    assert [segment.clause_id for segment in merged] == ["c-1", "c-2"]
    assert merged[0].section_ref == "§9"


def test_segment_new_identity_appends() -> None:
    merged = merge_segments([_segment("c-1")], [_segment("c-3")])

    assert [segment.clause_id for segment in merged] == ["c-1", "c-3"]


def test_segment_double_replay_is_idempotent() -> None:
    once = merge_segments([], [_segment("c-1")])
    twice = merge_segments(once, [_segment("c-1")])

    assert twice == once


def test_entity_merge_keys_on_entity_id() -> None:
    merged = merge_entities([_entity("e-1")], [_entity("e-1"), _entity("e-2")])

    assert [entity.entity_id for entity in merged] == ["e-1", "e-2"]


def test_relationship_merge_keys_on_edge_id_and_replay_wins() -> None:
    merged = merge_relationships([_relationship("edge-1")], [_relationship("edge-1", 0.7)])

    assert len(merged) == 1
    assert merged[0].citation.confidence == pytest.approx(0.7)
