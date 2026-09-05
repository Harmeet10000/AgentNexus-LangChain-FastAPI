"""Unit tests for Band D item 1 — persistence retarget onto `documents`/`chunks`.

D15 (`documents`/`chunks` is the sole retrieval schema, Accepted) forbids creating
`parent_documents`, `entities`, `relationships`, and `clauses`: none exists and no
migration will create them. The promoted pipeline's persistence stages issued raw
SQL against all four. These tests pin the retarget:

- no persistence statement in `ingestion_kb/nodes.py` names a forbidden relation;
- the document upsert targets `documents` with the (user_id, content_hash)
  identity and the writer-supplied `id` / `object_uri` D15 requires;
- chunk writes target `chunks` with the (document_id, chunk_index) upsert key,
  `clause` as a `chunk_kind` value, and `updated_at` maintained on both the
  insert and the conflict path (the D15 trap);
- entity/relationship stages issue no SQL at all — the graph-episode path is the
  sole writer — while still resolving in-memory identities downstream stages use.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from app.shared.langgraph_layer.ingestion_kb import nodes as nodes_module
from app.shared.langgraph_layer.ingestion_kb.nodes import (
    _store_chunks,
    _store_entities,
    _store_relationships,
    _upsert_parent_document,
)
from app.shared.langgraph_layer.ingestion_kb.state import (
    ClauseType,
    ContextualizedChunk,
    ContractMetadata,
    EntityType,
    ExtractedEntity,
    ExtractedRelationship,
    IngestionState,
    ParsedDocument,
    RelationType,
)

if TYPE_CHECKING:
    from typing import Any

_NODES_SOURCE = Path(nodes_module.__file__).read_text(encoding="utf-8")

# Relations D15 forbids creating. `entities`/`relationships`/`clauses` are matched
# with their statement keyword so prose comments naming them do not fail the gate;
# `parent_documents` has no legitimate mention anywhere in this module.
_FORBIDDEN_STATEMENTS = (
    "INSERT INTO parent_documents",
    "INSERT INTO entities",
    "INSERT INTO relationships",
    "INSERT INTO clauses",
    "parent_documents",
    "clauses_bm25_idx",
)


def test_no_persistence_statement_names_a_forbidden_relation() -> None:
    for statement in _FORBIDDEN_STATEMENTS:
        assert statement not in _NODES_SOURCE, statement


def test_the_chunk_table_is_the_only_relational_write_target() -> None:
    assert "INSERT INTO documents" in _NODES_SOURCE
    assert "INSERT INTO chunks" in _NODES_SOURCE


class _FakeResult:
    def __init__(self, row: tuple[object, ...] | None) -> None:
        self._row = row

    def fetchone(self) -> tuple[object, ...] | None:
        return self._row


class _FakeSession:
    """Captures raw-SQL executions without a database."""

    def __init__(self, row: tuple[object, ...] | None = ("row-id",)) -> None:
        self.statements: list[str] = []
        self.params: list[dict[str, Any]] = []
        self._row = row

    async def execute(
        self, query: Any, params: dict[str, Any] | None = None
    ) -> _FakeResult:
        self.statements.append(str(query))
        self.params.append(dict(params or {}))
        return _FakeResult(self._row)


def _state(**overrides: Any) -> IngestionState:
    values: dict[str, Any] = {
        "doc_id": "doc-1",
        "user_id": "user-1",
        "thread_id": "thread-1",
        "source": "s3://bucket/key",
        "filename": "contract.pdf",
    }
    values.update(overrides)
    return IngestionState(**values)


def _parsed(**overrides: Any) -> ParsedDocument:
    values: dict[str, Any] = {
        "markdown": "# Agreement\n\nBody text here.",
        "title": "Agreement",
        "source": "s3://bucket/key",
    }
    values.update(overrides)
    return ParsedDocument(**values)


def _metadata(**overrides: Any) -> ContractMetadata:
    values: dict[str, Any] = {
        "contract_name": "MSA",
        "parties": ["Acme Inc", "Beta LLC"],
        "document_summary": "A master services agreement.",
    }
    values.update(overrides)
    return ContractMetadata(**values)


def _chunk(**overrides: Any) -> ContextualizedChunk:
    values: dict[str, Any] = {
        "clause_id": "clause-1",
        "chunk_index": 0,
        "clause_type": ClauseType.INDEMNITY,
        "preamble": "This is indemnity from MSA.",
        "text": "The Supplier shall indemnify.",
        "tokens": 12,
        "page_no": 1,
    }
    values.update(overrides)
    return ContextualizedChunk(**values)


async def test_parent_upsert_targets_documents_with_identity_conflict() -> None:
    session = _FakeSession(row=("doc-uuid-1",))
    state = _state()
    document_id = await _upsert_parent_document(session, state, _parsed(), _metadata())

    assert document_id == "doc-uuid-1"
    assert len(session.statements) == 1
    sql = session.statements[0]
    assert "INSERT INTO documents" in sql
    assert "uq_documents_user_content_hash" in sql
    # The full-text body must not be persisted: `documents` carries no body
    # column, and the markdown must not be smuggled into `metadata_`.
    assert "markdown" not in sql.lower()

    params = session.params[0]
    assert params["user_id"] == "user-1"
    assert params["title"] == "Agreement"
    assert params["object_uri"] == "s3://bucket/key"
    assert params["id"]  # writer-supplied: no database default exists
    assert len(params["content_hash"]) == 64
    assert params["status"] == "processing"
    stored_metadata = json.loads(params["metadata_"])
    assert stored_metadata["thread_id"] == "thread-1"
    assert stored_metadata["document_summary"] == "A master services agreement."


async def test_parent_upsert_raises_when_no_id_returns() -> None:
    session = _FakeSession(row=None)
    with pytest.raises(ValueError, match="did not return an id"):
        await _upsert_parent_document(session, _state(), _parsed(), _metadata())


async def test_object_uri_is_never_empty() -> None:
    session = _FakeSession(row=("doc-uuid-9",))
    state = _state(doc_id="doc-9", source="", filename="")
    await _upsert_parent_document(session, state, _parsed(source=""), _metadata())
    assert session.params[0]["object_uri"] == "ingest://doc-9"


async def test_store_chunks_targets_chunks_with_upsert_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_embed(
        texts: list[str], *, task_type: Any = None, redis: Any = None
    ) -> list[list[float]]:
        assert len(texts) == 2
        return [[0.1, 0.2]] * len(texts)

    monkeypatch.setattr(nodes_module, "embed_texts", _fake_embed)
    session = _FakeSession(row=("chunk-uuid-1",))
    state = _state(contextualized_chunks=[_chunk(chunk_index=0), _chunk(chunk_index=1)])

    stored = await _store_chunks(
        session=session,
        state=state,
        parsed=_parsed(),
        metadata=_metadata(),
        parent_doc_id="doc-uuid-1",
        redis=None,
    )

    assert [item.chunk_id for item in stored] == ["chunk-uuid-1", "chunk-uuid-1"]
    assert len(session.statements) == 2
    for sql in session.statements:
        assert "INSERT INTO chunks" in sql
        assert "uq_chunks_document_chunk_index" in sql
        # `search_text` is generated by the database and must never be supplied.
        assert "search_text" not in sql
        # The D15 trap: `updated_at` maintained on both the insert and the
        # conflict-resolved update.
        assert sql.count("updated_at") >= 2

    params = session.params[0]
    assert params["document_id"] == "doc-uuid-1"
    assert params["user_id"] == "user-1"
    assert params["chunk_index"] == 0
    assert params["chunk_kind"] == "clause"
    assert params["clause_type"] == ClauseType.INDEMNITY.value
    assert params["id"]  # writer-supplied: no database default exists


async def test_store_chunks_writes_nothing_without_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _boom(
        texts: list[str], *, task_type: Any = None, redis: Any = None
    ) -> list[list[float]]:
        msg = "must not embed when there is nothing to store"
        raise AssertionError(msg)

    monkeypatch.setattr(nodes_module, "embed_texts", _boom)
    session = _FakeSession()
    stored = await _store_chunks(
        session=session,
        state=_state(),
        parsed=_parsed(),
        metadata=_metadata(),
        parent_doc_id="doc-uuid-1",
        redis=None,
    )
    assert stored == []
    assert session.statements == []


async def test_entities_and_relationships_issue_no_sql() -> None:
    session = _FakeSession()
    state = _state(
        extracted_entities=[
            ExtractedEntity(
                id="e1", type=EntityType.ORG, name="Acme Inc.", normalized_name="Acme Inc"
            ),
            ExtractedEntity(
                id="e2", type=EntityType.ORG, name="Beta", normalized_name="Beta"
            ),
        ],
        extracted_relationships=[
            ExtractedRelationship(
                from_entity="e1",
                to_entity="e2",
                type=RelationType.SIGNED_BY,
                clause_id="clause-1",
            ),
            ExtractedRelationship(
                from_entity="e1",
                to_entity="missing",
                type=RelationType.SIGNED_BY,
            ),
        ],
    )

    entity_map = await _store_entities(session, state)
    relationships = await _store_relationships(session, state, entity_map)

    assert session.statements == []
    assert set(entity_map) == {"e1", "e2"}
    # One relationship resolves; the one naming a refused endpoint is skipped.
    assert len(relationships) == 1
    assert "SIGNED_BY" in relationships[0]
