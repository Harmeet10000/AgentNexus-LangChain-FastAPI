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

from typing import TYPE_CHECKING

from langchain_core.tools import tool
from sqlalchemy import text

from app.utils import logger

from .idempotency import IdempotencyGuard, ToolResult

if TYPE_CHECKING:
    from typing import Any

    from langchain_core.tools.base import BaseTool
    from sqlalchemy.ext.asyncio import AsyncEngine

    from app.shared.rag.graphiti.client import GraphitiService

_MIN_SOURCE_THRESHOLD: int = 2
_STATUTE_SEARCH_LIMIT: int = 5


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
            input_data={"query": query, "clause_id": clause_id, "jurisdiction": jurisdiction},
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
        statute_results = await _search_statutes_postgres(
            db_engine=db_engine,
            query=query,
            jurisdiction=jurisdiction,
            limit=_STATUTE_SEARCH_LIMIT,
        )

        total_sources = len(graphiti_results) + len(statute_results)
        insufficient_basis = total_sources < _MIN_SOURCE_THRESHOLD

        if insufficient_basis:
            log.warning(
                "precedent_insufficient_basis",
                total_sources=total_sources,
                clause_id=clause_id,
            )

        result = ToolResult.ok(
            data={
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
    """Full-text search against the statutes table.

    Assumes schema:
        statutes (
            id          UUID PRIMARY KEY,
            title       TEXT,
            section_ref VARCHAR(64),
            body        TEXT,
            jurisdiction VARCHAR(128),
            act_name    VARCHAR(255),
            year        INT,
            fts_vector  TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', body)) STORED
        )
        CREATE INDEX ON statutes USING gin(fts_vector);

    Falls back to empty list on schema-not-found — lets you deploy before
    the statutes table is populated.
    """
    query_sql = text(
        """
        SELECT
            id::text,
            title,
            section_ref,
            LEFT(body, 500) AS excerpt,
            jurisdiction,
            act_name,
            year,
            ts_rank(fts_vector, plainto_tsquery('english', :query)) AS rank
        FROM statutes
        WHERE
            jurisdiction ILIKE :jurisdiction
            AND fts_vector @@ plainto_tsquery('english', :query)
        ORDER BY rank DESC
        LIMIT :limit
        """
    )
    try:
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
                    "jurisdiction": row[4],
                    "act_name": row[5],
                    "year": row[6],
                    "rank": float(row[7]),
                    "source": "postgres_statutes",
                }
                for row in rows
            ]
    except Exception as exc:
        logger.warning("statute_postgres_search_failed", error=str(exc))
        return []
