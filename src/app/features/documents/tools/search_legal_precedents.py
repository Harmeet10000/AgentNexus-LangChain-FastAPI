"""
Tool: search_legal_precedents

Compliance agent tool.  Retrieval-first — no hallucinations allowed.

Data sources (Section 8.7 — both Postgres and Graphiti):
  1. Graphiti: precedent chains from user's prior approved documents
     → how similar clauses were handled before, cross-document patterns
  2. Postgres statutes table: exact statute applicability lookups
     → statute text, section numbers, jurisdiction metadata

Guardrail (Section 8.7):
  If total_sources < _MIN_SOURCE_THRESHOLD:
      return "Insufficient legal basis" signal in ToolResult

The tool is idempotency-guarded.  The LLM cannot distinguish a cached
result from a live retrieval — nor should it.  Determinism is the goal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from langchain.tools import ToolRuntime
from langchain_core.tools import InjectedToolArg, tool
from langchain_core.tools.base import BaseTool
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.features.documents.constants import (
    HYBRID_CANDIDATE_LIMIT,
    RRF_K,
    TRIGRAM_SIMILARITY_THRESHOLD,
)
from app.kernel.result import add_database_error_note
from app.observability.logging import logger
from app.resilience.idempotency import IdempotencyGuard, ToolResult
from app.retrieval.fusion import RankedResultRow, reciprocal_rank_fusion
from app.retrieval.graphiti.schemas import GraphitiAvailability

from .authorization import trusted_tool_context

_MIN_SOURCE_THRESHOLD: int = 2
_STATUTE_SEARCH_LIMIT: int = 5

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from app.retrieval.graphiti.client import BoundGraphitiService as GraphitiService


def make_search_legal_precedents_tool(
    graphiti_service: GraphitiService,
    db_engine: AsyncEngine,
    idempotency: IdempotencyGuard,
) -> BaseTool:
    """Factory: returns @tool with injected infra via closure.

    Call once at lifespan startup.  Store result in ToolRegistry.
    """

    @tool
    async def search_legal_precedents(
        query: str,
        clause_id: str,
        jurisdiction: str,
        *,
        runtime: Annotated[ToolRuntime | None, InjectedToolArg] = None,
        user_id: Annotated[str | None, InjectedToolArg] = None,
        thread_id: Annotated[str | None, InjectedToolArg] = None,
        step_id: Annotated[str | None, InjectedToolArg] = None,
    ) -> dict[str, Any]:
        """Search for legal precedents and statutes relevant to a clause.

        Combines Graphiti knowledge graph (prior approved documents)
        with Postgres statutes table (authoritative legal text).

        Returns sources with citations.  If fewer than 2 sources found,
        sets insufficient_basis=True — the compliance agent MUST NOT
        make a determination without sufficient legal basis.

        Args:
            query: Natural language search query (e.g. 'limitation of liability India')
            clause_id: Clause being analysed (for idempotency key)
            jurisdiction: Target jurisdiction (default: India)
            Runtime identity and document scope are injected by LangGraph and
            are intentionally absent from the model-facing schema.
        """
        authz = trusted_tool_context(
            runtime,
            legacy_user_id=user_id,
            legacy_thread_id=thread_id,
            legacy_step_id=step_id,
        )
        authz.require_operation("evidence.search")
        log = logger.bind(tool="search_legal_precedents", clause_id=clause_id)

        idem_key = IdempotencyGuard.make_key(
            step_id=authz.step_id,
            structural={"clause_id": clause_id, "jurisdiction": jurisdiction},
            content={"query": query},
            user_id=authz.user_id,
            tool_name="search_legal_precedents",
            tenant_id=authz.tenant_id,
            allowed_doc_ids=authz.allowed_doc_ids,
            evidence_policy="mechanically_verified",
            document_versions=authz.document_versions,
            thread_id=authz.thread_id,
            run_id=authz.run_id,
            checkpoint_id=authz.checkpoint_id,
        )
        cached = await idempotency.get(idem_key)
        if cached is not None:
            log.debug("precedent_search_cache_hit")
            return cached.model_dump()

        # --- Graphiti: precedent chains from prior reviewed documents -------
        graphiti_results = await graphiti_service.search_for_precedent_chains(
            query=query,
            user_id=authz.user_id,
            allowed_doc_ids=authz.allowed_doc_ids,
            jurisdiction=jurisdiction,
            num_results=5,
            tenant_id=authz.tenant_id,
            document_versions=tuple(
                (document_id, version) for document_id, version, _ in authz.document_versions
            ),
            document_contracts=authz.document_contracts,
        )

        # --- Postgres: statute text retrieval --------------------------------
        # Contract (agent-tool-contract): when one leg is unreachable the
        # sufficiency verdict must NOT be computed from the surviving source
        # alone — expose basis-unknown and keep whatever the other leg found.
        unavailable_layers: list[str] = []
        graphiti_availability = getattr(
            graphiti_results, "availability", GraphitiAvailability.AVAILABLE
        )
        if graphiti_availability == GraphitiAvailability.UNAVAILABLE:
            unavailable_layers.append("graphiti")
        try:
            statute_results = await _search_statutes_postgres(
                db_engine=db_engine,
                query=query,
                jurisdiction=jurisdiction,
                limit=_STATUTE_SEARCH_LIMIT,
                tenant_id=authz.tenant_id,
                allowed_doc_ids=authz.allowed_doc_ids,
            )
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="statute_sections", operation="search_statutes")
            log.exception("statute_postgres_search_failed")
            unavailable_layers.append("statutes")
            statute_results = []

        basis_unknown = bool(unavailable_layers)
        if unavailable_layers and not graphiti_results and not statute_results:
            # Every leg failed or answered empty while one was unreachable:
            # report unavailability rather than a fabricated "no results".
            unavailable = ToolResult.unavailable_result(
                reason="no precedent layer available: " + ", ".join(unavailable_layers),
                clause_id=clause_id,
            )
            return unavailable.model_dump()

        total_sources = len(graphiti_results) + len(statute_results)
        insufficient_basis: bool | None = (
            None if basis_unknown else total_sources < _MIN_SOURCE_THRESHOLD
        )

        if insufficient_basis:
            log.warning(
                "precedent_insufficient_basis",
                total_sources=total_sources,
                clause_id=clause_id,
            )

        result = ToolResult.ok(
            data={
                "unavailable_layers": unavailable_layers,
                "basis_unknown": basis_unknown,
                "graphiti_availability": str(graphiti_availability),
                "graphiti_coverage": getattr(graphiti_results, "coverage", {}),
                "precedents": [
                    {
                        "name": r.name,
                        "content": r.content,
                        "relevance_score": r.relevance_score,
                        "source": "graphiti_knowledge_graph",
                    }
                    for r in graphiti_results
                ],
                "statutes": statute_results,
                "total_sources": total_sources,
                "insufficient_basis": insufficient_basis,
                "jurisdiction": jurisdiction,
            },
            tool="search_legal_precedents",
            clause_id=clause_id,
        )

        await idempotency.set(
            key=idem_key,
            result=result,
            tool_name="search_legal_precedents",
            user_id=authz.tenant_id,
            thread_id=authz.thread_id,
            step_id=authz.step_id,
        )

        log.info(
            "precedent_search_complete",
            total_sources=total_sources,
            insufficient_basis=insufficient_basis,
        )
        return result.model_dump()

    return search_legal_precedents


async def _search_statutes_postgres(
    db_engine: AsyncEngine,
    query: str,
    jurisdiction: str,  # noqa: ARG001 — kept in the tool's contract; the statute corpus is jurisdiction-agnostic until change 2 annotates it
    limit: int,
    *,
    tenant_id: str,
    allowed_doc_ids: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Full-text statute search over the unified chunk corpus.

    Harvested from the shared retrieval path, not a second ranking
    implementation (D-11):

    - BM25 branch reuses ``DocumentRepository.bm25_search``'s ranking
      expression verbatim — ``search_text <@> to_bm25query(:query,
      'chunks_bm25_idx')`` — including its ``< 0`` match predicate. The index
      name stays a literal at this use site per the constants-module contract
      (Decision 10: identifiers are asserted by the schema gate, never
      interpolated).
    - Trigram branch reuses ``DocumentRepository.trigram_search``'s
      ``similarity`` expression and threshold.
    - The two ranked lists fuse through the shared
      ``reciprocal_rank_fusion`` (``k=RRF_K``), the same function
      ``DocumentService`` fuses its branches with.

    Both branches scope to statute chunks and the server-issued document
    allowlist. An empty allowlist fails closed.
    ``SQLAlchemyError`` propagates to the caller, which maps it onto
    ``ToolResult.unavailable_result`` / ``basis_unknown`` — this helper never
    swallows reachability into an empty list.
    """
    if not allowed_doc_ids:
        return []

    bm25_sql = text(
        """
        SELECT
            c.id::text AS chunk_id,
            ROW_NUMBER() OVER (
                ORDER BY c.search_text <@> to_bm25query(:query, 'chunks_bm25_idx')
            ) AS rank
        FROM chunks AS c
        WHERE
            c.instrument_name IS NOT NULL
            AND c.tenant_id = :tenant_id
            AND c.document_id = ANY(CAST(:allowed_doc_ids AS uuid[]))
            AND (c.search_text <@> to_bm25query(:query, 'chunks_bm25_idx')) < 0
        ORDER BY c.search_text <@> to_bm25query(:query, 'chunks_bm25_idx')
        LIMIT :candidate_limit
        """
    )
    trigram_sql = text(
        """
        SELECT
            c.id::text AS chunk_id,
            ROW_NUMBER() OVER (ORDER BY similarity(c.search_text, :query) DESC) AS rank
        FROM chunks AS c
        WHERE
            c.instrument_name IS NOT NULL
            AND c.tenant_id = :tenant_id
            AND c.document_id = ANY(CAST(:allowed_doc_ids AS uuid[]))
            AND c.search_text % :query
            AND similarity(c.search_text, :query) >= :similarity_threshold
        ORDER BY similarity(c.search_text, :query) DESC
        LIMIT :candidate_limit
        """
    )
    detail_sql = text(
        """
        SELECT
            id::text,
            instrument_name AS title,
            section_ref,
            LEFT(content, 500) AS excerpt,
            document_id::text AS document_id,
            instrument_name AS act_name,
            instrument_year AS year
        FROM chunks
        WHERE id = ANY(CAST(:chunk_ids AS uuid[]))
          AND tenant_id = :tenant_id
          AND document_id = ANY(CAST(:allowed_doc_ids AS uuid[]))
        """
    )
    async with db_engine.connect() as conn:
        bm25_rows = (
            await conn.execute(
                bm25_sql,
                {
                    "query": query,
                    "candidate_limit": HYBRID_CANDIDATE_LIMIT,
                    "tenant_id": tenant_id,
                    "allowed_doc_ids": list(allowed_doc_ids),
                },
            )
        ).fetchall()
        trigram_rows = (
            await conn.execute(
                trigram_sql,
                {
                    "query": query,
                    "candidate_limit": HYBRID_CANDIDATE_LIMIT,
                    "similarity_threshold": TRIGRAM_SIMILARITY_THRESHOLD,
                    "tenant_id": tenant_id,
                    "allowed_doc_ids": list(allowed_doc_ids),
                },
            )
        ).fetchall()

        fused = reciprocal_rank_fusion(
            [
                RankedResultRow(chunk_id=str(row[0]), score=0.0, rank=int(row[1]))
                for row in bm25_rows
            ],
            [
                RankedResultRow(chunk_id=str(row[0]), score=0.0, rank=int(row[1]))
                for row in trigram_rows
            ],
            k=RRF_K,
            limit=limit,
        )
        if not fused:
            return []

        score_by_id = {item.chunk_id: item.score for item in fused}
        details = (
            await conn.execute(
                detail_sql,
                {
                    "chunk_ids": [item.chunk_id for item in fused],
                    "tenant_id": tenant_id,
                    "allowed_doc_ids": list(allowed_doc_ids),
                },
            )
        ).fetchall()
        by_id = {str(row[0]): row for row in details}
        return [
            {
                "id": str(by_id[item.chunk_id][0]),
                "title": by_id[item.chunk_id][1],
                "section_ref": by_id[item.chunk_id][2],
                "excerpt": by_id[item.chunk_id][3],
                "jurisdiction": None,
                "act_name": by_id[item.chunk_id][5],
                "year": by_id[item.chunk_id][6],
                "rank": float(score_by_id[item.chunk_id]),
                "source": "unified_corpus",
            }
            for item in fused
            if item.chunk_id in by_id
        ]
