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

from typing import TYPE_CHECKING, Any

from langchain_core.tools import tool
from langchain_core.tools.base import BaseTool
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.shared.result.diagnostics import add_database_error_note
from app.utils import logger

from .idempotency import IdempotencyGuard, ToolResult

_MIN_SOURCE_THRESHOLD: int = 2
_STATUTE_SEARCH_LIMIT: int = 5

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from app.shared.rag.graphiti.client import GraphitiService


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
        user_id: str,
        thread_id: str,
        step_id: str,
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
            user_id: Current user ID (for Graphiti namespace scoping)
            thread_id: Current thread ID (for idempotency audit)
            step_id: Plan step ID (for idempotency key)
        """
        log = logger.bind(tool="search_legal_precedents", clause_id=clause_id)

        idem_key = IdempotencyGuard.make_key(
            step_id=step_id,
            structural={"clause_id": clause_id, "jurisdiction": jurisdiction},
            content={"query": query},
            user_id=user_id,
        )
        cached = await idempotency.get(idem_key)
        if cached is not None:
            log.debug("precedent_search_cache_hit")
            return cached.model_dump()

        # --- Graphiti: precedent chains from prior reviewed documents -------
        graphiti_results = await graphiti_service.search_for_precedent_chains(
            query=query,
            user_id=user_id,
            jurisdiction=jurisdiction,
            num_results=5,
        )

        # --- Postgres: statute text retrieval --------------------------------
        # Contract (agent-tool-contract): when one leg is unreachable the
        # sufficiency verdict must NOT be computed from the surviving source
        # alone — expose basis-unknown and keep whatever the other leg found.
        unavailable_layers: list[str] = []
        try:
            statute_results = await _search_statutes_postgres(
                db_engine=db_engine,
                query=query,
                jurisdiction=jurisdiction,
                limit=_STATUTE_SEARCH_LIMIT,
            )
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="statute_sections", operation="search_statutes")
            logger.warning("statute_postgres_search_failed", error=str(exc))
            unavailable_layers.append("statutes")
            statute_results = []

        basis_unknown = bool(unavailable_layers)
        if unavailable_layers and not graphiti_results:
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
            user_id=user_id,
            thread_id=thread_id,
            step_id=step_id,
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
    jurisdiction: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Full-text statute search over the unified chunk corpus.

    Task 11.1: the superseded `statutes` relation is created by no migration;
    retrieval resolves against `chunks.search_text`, which change 0 indexes
    (trgm GIN) and ranks by trigram similarity.
    """
    query_sql = text(
        """
        SELECT
            id::text,
            instrument_name AS title,
            section_ref,
            LEFT(content, 500) AS excerpt,
            document_id::text AS document_id,
            instrument_name AS act_name,
            instrument_year AS year,
            similarity(search_text, :query) AS rank
        FROM chunks
        WHERE
            instrument_name IS NOT NULL
            AND search_text %% :query ::text
        ORDER BY rank DESC
        LIMIT :limit
        """
    )
    async with db_engine.connect() as conn:
        rows = (
            await conn.execute(
                query_sql,
                {
                    "query": query,
                    "jurisdiction": f"%{jurisdiction}%",
                    "limit": limit,
                },
            )
        ).fetchall()
        return [
            {
                "id": str(row[0]),
                "title": row[1],
                "section_ref": row[2],
                "excerpt": row[3],
                "jurisdiction": None,
                "act_name": row[5],
                "year": row[6],
                "rank": float(row[7]),
                "source": "unified_corpus",
            }
            for row in rows
        ]
