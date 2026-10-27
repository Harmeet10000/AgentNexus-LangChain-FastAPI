"""Regression coverage for policy separation and chunk provenance."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from graphiti_core.errors import GraphitiError
from sqlalchemy.dialects import postgresql

from app.ai.graph.retry import TransientExternalError
from app.features.documents.dto import UnifiedAskResponse
from app.features.documents.repository import build_chunk_rows, build_chunk_upsert_statement
from app.features.documents.service.query import (
    _build_answer_cache_key,
    _graphiti_filter_chunk_ids,
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

    result = await _graphiti_filter_chunk_ids(
        graphiti=_OutageGraphiti(),  # type: ignore[arg-type]
        user_id="alice",
        query="termination",
        doc_ids_filter=["doc-a"],
    )

    # An empty optional graph narrowing must not turn a healthy Postgres search
    # into an empty answer.  Legal mode still applies its independent
    # graphiti_verified SQL predicate.
    assert result == []


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
