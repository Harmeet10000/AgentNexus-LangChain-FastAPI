"""Regression coverage for policy separation and chunk provenance."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from graphiti_core.errors import GraphitiError
from sqlalchemy.dialects import postgresql

from app.ai.graph.retry import TransientExternalError
from app.features.documents.adapters.persistence.postgres.document_repository import (
    build_chunk_rows,
    build_chunk_upsert_statement,
)
from app.features.documents.api.http.dto import UnifiedAskResponse
from app.features.documents.application.queries.documents import _build_answer_cache_key
from app.features.documents.application.retrieval.graph.nodes import make_graph_retrieval_node
from app.features.documents.application.retrieval.graph.state import (
    QueryPlan,
    RetrievalGraphContext,
)


def _answer_key(*, require_graphiti_verified: bool) -> str:
    return _build_answer_cache_key(
        user_id="alice",
        tenant_id="tenant-a",
        query="same question",
        doc_ids_filter=["doc-a"],
        jurisdiction=None,
        contract_type=None,
        clause_type=None,
        require_graphiti_verified=require_graphiti_verified,
    )


def test_ask_response_defaults_to_mechanically_verified_policy() -> None:
    response = UnifiedAskResponse(answer="answer", citations=[], confidence="uncertain")

    assert response.evidence_policy == "mechanically_verified"
    assert response.coverage.documents_requested == 0


def test_answer_cache_keys_are_separate_for_evidence_policies() -> None:
    assert _answer_key(require_graphiti_verified=False) != _answer_key(
        require_graphiti_verified=True
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [GraphitiError("outage"), TransientExternalError("outage")])
async def test_graphiti_filter_degrades_to_unfiltered_postgres_candidates(error: Exception) -> None:
    class _OutageGraphiti:
        async def search(self, **_: object) -> list[object]:
            raise error

    # Retargeted to the graph node (mount-retrieval-graph): the inline
    # `_graphiti_filter_chunk_ids` helper was deleted with the inline ask body,
    # and the optional graph narrowing now lives in the `graph_neo4j` node. The
    # degrade contract is unchanged: a graph outage empties the optional
    # narrowing instead of failing the request, so the Postgres legs still
    # answer. Legal mode still applies its independent graphiti_verified SQL
    # predicate.
    node = make_graph_retrieval_node()
    result = await node(
        {
            "user_id": "alice",
            "tenant_id": "tenant-a",
            "allowed_doc_ids": ["doc-a"],
            "scope_bound": True,
            "document_versions": [("doc-a", 1, None)],
            "document_contracts": [("doc-a", 1, (None,), True)],
            "scope_unavailable_documents": [],
            "contract_index": None,
            "doc_ids_filter": ["doc-a"],
            "query_plan": QueryPlan(rewritten_query="termination"),
        },
        SimpleNamespace(
            context=RetrievalGraphContext(graphiti=_OutageGraphiti())  # type: ignore[arg-type]
        ),
    )

    assert result == {"graph_chunk_ids": []}


def test_chunk_rows_and_upsert_carry_explicit_provenance() -> None:
    indexed_at = datetime(2026, 10, 27, tzinfo=UTC)
    rows = build_chunk_rows(
        document_id="00000000-0000-0000-0000-000000000001",
        user_id="alice",
        tenant_id="tenant-a",
        chunks=[
            {
                "content": "clause",
                "chunk_index": 0,
                "document_version": 2,
                "pipeline_version": "documents-v2",
                "graphiti_verified": True,
                "graph_indexed_at": indexed_at,
            }
        ],
    )
    row = rows[0]
    sql = str(build_chunk_upsert_statement(rows).compile(dialect=postgresql.dialect()))

    assert row["pipeline_version"] == "documents-v2"
    assert row["graph_indexed_at"] == indexed_at
    assert "pipeline_version" in sql
    assert "graph_indexed_at" in sql
