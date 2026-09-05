"""Unit tests for Band D item 2 — ADR-2 entity canonicalisation.

ADR-2 is the only irreversible contract in the ingestion change: once variant
surface forms become separate graph nodes, no later pass can merge them. These
tests pin both halves — variant forms collapse, and genuinely distinct parties
do **not** collide (the harder half, and the easier one to get wrong in the
unsafe direction) — plus determinism, raw-form retention, the refusal path, and
that both graph-reaching stages refuse before writing anything.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.shared.langgraph_layer.ingestion_kb.canonicalize import (
    canonical_identity_key,
    canonicalize_entities,
)
from app.shared.langgraph_layer.ingestion_kb.nodes import (
    _store_entities,
    make_graphiti_upsert_node,
)
from app.shared.langgraph_layer.ingestion_kb.state import (
    ContextualizedChunk,
    EntityType,
    ExtractedEntity,
    IngestionState,
)

if TYPE_CHECKING:
    from typing import Any


def _entity(
    name: str, ref: str = "e1", entity_type: EntityType = EntityType.ORG
) -> ExtractedEntity:
    return ExtractedEntity(
        id=ref, type=entity_type, name=name, normalized_name=name.strip().lower()
    )


def test_variant_surface_forms_resolve_to_one_identity() -> None:
    variants = (
        "Acme Inc",
        "ACME INC.",
        "Acme, Incorporated",
        "the acme co",
        "Acme & Co",
        "Acme and Company",
    )
    keys = {canonical_identity_key(entity_type="ORG", name=v) for v in variants}
    assert len(keys) == 1


def test_distinct_parties_do_not_collide() -> None:
    acme_key = canonical_identity_key(entity_type="ORG", name="Acme Inc")
    assert acme_key != canonical_identity_key(entity_type="ORG", name="Beta LLC")
    assert acme_key != canonical_identity_key(entity_type="ORG", name="Acmebet Inc")
    # Same stem under different entity types stays apart: a person and the
    # company they are named after are not one node.
    assert canonical_identity_key(
        entity_type="ORG", name="Acme"
    ) != canonical_identity_key(entity_type="PERSON", name="Acme")


def test_canonicalisation_is_deterministic() -> None:
    first = canonical_identity_key(entity_type="ORG", name="Acme Holdings Group, Inc.")
    second = canonical_identity_key(entity_type="ORG", name="Acme Holdings Group, Inc.")
    assert first == second == "ORG:acme"


def test_raw_surface_form_is_retained() -> None:
    canonical, refused = canonicalize_entities([_entity("Acme Inc.")])
    assert refused == []
    record = canonical["e1"]
    assert record.raw_name == "Acme Inc."
    assert record.canonical_id == "ORG:acme"
    assert record.entity_type == "ORG"


def test_uncanonicalisable_entities_are_refused_not_fallback() -> None:
    entities = [
        _entity("", ref="blank"),
        _entity("   ", ref="spaces"),
        _entity("!!!", ref="punct"),
        _entity("LLC", ref="suffix"),
        _entity("Acme Inc", ref="good"),
    ]
    canonical, refused = canonicalize_entities(entities)
    assert {entity.id for entity in refused} == {"blank", "spaces", "punct", "suffix"}
    assert set(canonical) == {"good"}


class _FakeSession:
    """Records raw-SQL executions without a database."""

    def __init__(self) -> None:
        self.statements: list[str] = []

    async def execute(self, query: Any, params: Any = None) -> Any:
        self.statements.append(str(query))
        return None


async def test_store_entities_skips_refused_without_sql() -> None:
    session = _FakeSession()
    state = IngestionState(
        extracted_entities=[_entity("!!!", ref="bad"), _entity("Acme Inc", ref="good")]
    )
    entity_map = await _store_entities(session, state)
    assert session.statements == []
    assert set(entity_map) == {"good"}
    assert entity_map["good"] == "ORG:acme"


async def test_graphiti_upsert_refuses_before_any_episode() -> None:
    calls: list[dict[str, Any]] = []

    class _Graphiti:
        async def add_episode(self, **kwargs: Any) -> Any:
            calls.append(kwargs)
            msg = "must not write when canonicalisation refuses"
            raise AssertionError(msg)

    node = make_graphiti_upsert_node(_Graphiti())
    state = IngestionState(
        doc_id="doc-1",
        extracted_entities=[_entity("...", ref="bad")],
        contextualized_chunks=[
            ContextualizedChunk(
                clause_id="clause-1",
                chunk_index=0,
                preamble="pre",
                text="body",
                tokens=3,
            )
        ],
    )
    result = await node(state)
    assert calls == []
    # A refusal returns the terminal-failure shape, which carries no episode
    # channel at all — not an empty episode list beside a failure.
    assert result.get("graphiti_episode_ids", []) == []
    assert result["ingestion_complete"] is False
    assert result["failure"] is not None


async def test_graphiti_upsert_carries_canonical_entity_references() -> None:
    seen: list[dict[str, Any]] = []

    class _Episode:
        uuid = "episode-1"

    class _Graphiti:
        async def add_episode(self, **kwargs: Any) -> _Episode:
            seen.append(kwargs)
            return _Episode()

    node = make_graphiti_upsert_node(_Graphiti())
    state = IngestionState(
        doc_id="doc-1",
        extracted_entities=[_entity("Acme Inc.")],
        contextualized_chunks=[
            ContextualizedChunk(
                clause_id="clause-1",
                chunk_index=0,
                preamble="pre",
                text="body",
                tokens=3,
            )
        ],
        stored_chunks=[],
    )
    # A stored chunk id is required for the episode to reference the clause row.
    from app.shared.langgraph_layer.ingestion_kb.state import StoredChunk

    state = state.model_copy(
        update={
            "stored_chunks": [
                StoredChunk(
                    chunk_id="chunk-uuid-1",
                    clause_id="clause-1",
                    chunk_index=0,
                    clause_type="indemnity",
                )
            ]
        }
    )
    result = await node(state)
    assert result["graphiti_episode_ids"] == ["episode-1"]
    assert result["ingestion_complete"] is True
    assert len(seen) == 1
    import json

    description = json.loads(seen[0]["source_description"])
    assert description["entities"] == ["ORG:acme"]


async def test_graphiti_upsert_skips_when_upstream_failed() -> None:
    from app.shared.langgraph_layer.ingestion_kb.errors import (
        IngestionGraphValidationError,
    )

    class _Graphiti:
        async def add_episode(self, **kwargs: Any) -> Any:
            msg = "must not write for a failed document"
            raise AssertionError(msg)

    node = make_graphiti_upsert_node(_Graphiti())
    state = IngestionState(
        doc_id="doc-1",
        failure=IngestionGraphValidationError(
            message="upstream refused", source="test"
        ),
    )
    result = await node(state)
    assert result["graphiti_episode_ids"] == []
    assert result["ingestion_complete"] is False
