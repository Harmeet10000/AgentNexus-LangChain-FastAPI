"""
Tool: retrieve_statute_section

Compliance agent tool.  Point lookup — NOT semantic search.

When the compliance agent already knows the statute and section
(e.g., "IT Act 2000, Section 43A"), use this tool for the exact text.

This is separate from search_legal_precedents deliberately:
  search_legal_precedents → discovery (don't know what applies yet)
  retrieve_statute_section → lookup   (know exactly what to fetch)

Idempotency: section lookups are pure reads — same input always returns
same output.  Cached in Redis for the session TTL, Postgres for 30 days.
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

def make_retrieve_statute_section_tool(
    db_engine: AsyncEngine,
    idempotency: IdempotencyGuard,
) -> BaseTool:

    @tool
    async def retrieve_statute_section(
        act_name: str,
        section_ref: str,
        jurisdiction: str,
        user_id: str,
        thread_id: str,
        step_id: str,
    ) -> dict[str, Any]:
        """Retrieve the exact text of a specific statute section.

        Use this when you already know the statute and section number.
        For discovery (don't know which statute applies), use search_legal_precedents.

        Args:
            act_name: Name of the act (e.g., 'Indian Contract Act 1872')
            section_ref: Section reference (e.g., '73', '43A', '2(h)')
            jurisdiction: Jurisdiction (e.g., 'India', 'India - Maharashtra')
            user_id: Current user ID
            thread_id: Current thread ID
            step_id: Plan step ID
        """
        log = logger.bind(tool="retrieve_statute_section", act=act_name, section=section_ref)

        idem_key = IdempotencyGuard.make_key(
            step_id=step_id,
            input_data={
                "act_name": act_name,
                "section_ref": section_ref,
                "jurisdiction": jurisdiction,
            },
            user_id=user_id,
        )
        cached = await idempotency.get(idem_key)
        if cached is not None:
            log.debug("statute_section_cache_hit")
            return cached.model_dump()

        row = await _fetch_statute_section(
            db_engine=db_engine,
            act_name=act_name,
            section_ref=section_ref,
            jurisdiction=jurisdiction,
        )

        if row is None:
            result = ToolResult.fail(
                error=f"Section {section_ref} of {act_name} not found in {jurisdiction}",
                act_name=act_name,
                section_ref=section_ref,
            )
        else:
            result = ToolResult.ok(
                data={
                    "id": row["id"],
                    "act_name": row["act_name"],
                    "section_ref": row["section_ref"],
                    "title": row["title"],
                    "body": row["body"],
                    "jurisdiction": row["jurisdiction"],
                    "year": row["year"],
                    "source": "postgres_statutes",
                },
                tool="retrieve_statute_section",
            )

        await idempotency.set(
            key=idem_key,
            result=result,
            tool_name="retrieve_statute_section",
            user_id=user_id,
            thread_id=thread_id,
            step_id=step_id,
        )
        log.info("statute_section_retrieved", found=result.success)
        return result.model_dump()

    return retrieve_statute_section


async def _fetch_statute_section(
    db_engine: AsyncEngine,
    act_name: str,
    section_ref: str,
    jurisdiction: str,
) -> dict[str, Any] | None:
    query = text(
        """
        SELECT
            id::text,
            act_name,
            section_ref,
            title,
            body,
            jurisdiction,
            year
        FROM statutes
        WHERE
            act_name ILIKE :act_name
            AND section_ref ILIKE :section_ref
            AND jurisdiction ILIKE :jurisdiction
        ORDER BY year DESC
        LIMIT 1
        """
    )
    try:
        async with db_engine.connect() as conn:
            row = (
                await conn.execute(
                    query,
                    {
                        "act_name": f"%{act_name}%",
                        "section_ref": section_ref.strip(),
                        "jurisdiction": f"%{jurisdiction}%",
                    },
                )
            ).fetchone()
            if row is None:
                return None
            return {
                "id": row[0],
                "act_name": row[1],
                "section_ref": row[2],
                "title": row[3],
                "body": row[4],
                "jurisdiction": row[5],
                "year": row[6],
            }
    except Exception as exc:
        logger.warning("statute_fetch_failed", error=str(exc))
        return None
