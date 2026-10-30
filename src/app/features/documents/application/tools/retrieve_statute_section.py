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

from typing import TYPE_CHECKING, Annotated, Any

from langchain.tools import ToolRuntime
from langchain_core.tools import InjectedToolArg, tool
from langchain_core.tools.base import BaseTool
from sqlalchemy import String, cast, select
from sqlalchemy.exc import SQLAlchemyError

from app.features.documents.domain.chunk import UnifiedChunk
from app.kernel.result import add_database_error_note
from app.observability.logging import logger
from app.resilience.idempotency import IdempotencyGuard, ToolResult
from app.utils.sql_binds import uuid_any

from .authorization import trusted_tool_context

if TYPE_CHECKING:
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
        *,
        runtime: Annotated[ToolRuntime | None, InjectedToolArg] = None,
        user_id: Annotated[str | None, InjectedToolArg] = None,
        thread_id: Annotated[str | None, InjectedToolArg] = None,
        step_id: Annotated[str | None, InjectedToolArg] = None,
    ) -> dict[str, Any]:
        """Retrieve the exact text of a specific statute section.

        Use this when you already know the statute and section number.
        For discovery (don't know which statute applies), use search_legal_precedents.

        Args:
            act_name: Name of the act (e.g., 'Indian Contract Act 1872')
            section_ref: Section reference (e.g., '73', '43A', '2(h)')
            jurisdiction: Jurisdiction (e.g., 'India', 'India - Maharashtra')
            Runtime identity and document scope are injected by LangGraph.
        """
        authz = trusted_tool_context(
            runtime,
            legacy_user_id=user_id,
            legacy_thread_id=thread_id,
            legacy_step_id=step_id,
        )
        authz.require_operation("evidence.read")
        log = logger.bind(tool="retrieve_statute_section", act=act_name, section=section_ref)

        idem_key = IdempotencyGuard.make_key(
            step_id=authz.step_id,
            structural={
                "act_name": act_name,
                "section_ref": section_ref,
                "jurisdiction": jurisdiction,
            },
            content=None,
            user_id=authz.user_id,
            tool_name="retrieve_statute_section",
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
            log.debug("statute_section_cache_hit")
            return cached.model_dump()

        try:
            row = await _fetch_statute_section(
                db_engine=db_engine,
                act_name=act_name,
                section_ref=section_ref,
                jurisdiction=jurisdiction,
                tenant_id=authz.tenant_id,
                allowed_doc_ids=authz.allowed_doc_ids,
                user_id=user_id if authz.legacy_compat else None,
            )
        except SQLAlchemyError as exc:
            add_database_error_note(
                exc, table="statute_sections", operation="fetch_statute_section"
            )
            # Honesty (group 6): an unreachable corpus is NOT a missing section.
            log.exception("statute_fetch_failed")
            unavailable = ToolResult.unavailable_result(
                reason=f"statute corpus unreachable ({type(exc).__name__})",
                act_name=act_name,
                section_ref=section_ref,
            )
            return unavailable.model_dump()

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
            user_id=authz.tenant_id,
            thread_id=authz.thread_id,
            step_id=authz.step_id,
        )
        log.info("statute_section_retrieved", found=result.success)
        return result.model_dump()

    return retrieve_statute_section


async def _fetch_statute_section(
    db_engine: AsyncEngine,
    act_name: str,
    section_ref: str,
    jurisdiction: str,  # noqa: ARG001 — kept in the tool's contract; corpus is jurisdiction-agnostic today
    *,
    tenant_id: str | None = None,
    allowed_doc_ids: tuple[str, ...] | None = None,
    user_id: str | None = None,
) -> dict[str, Any] | None:
    # Task 11.1: the point lookup targets the unified chunk corpus — the only
    # relations the migration history creates — via the statute identity
    # attributes (instrument_name / section_ref / instrument_year) carried on
    # chunks per revision f2a9c47b81de. The newest applicable year wins
    # (NULLS LAST). Exact equality is intentional: the predecessor used
    # ILIKE '%...%' prefix matching which could match the wrong statute
    # and could not be index-served. Corpus ingestion normalizes
    # instrument_name/section_ref (strip), and the LLM prompt constrains the
    # tool to canonical identifiers; case/prefix variants are thus a caller
    # contract, not a query concern. A functional LOWER index can be added
    # if case-variant traffic is observed.
    legacy_scope = tenant_id is not None and not allowed_doc_ids and user_id is not None
    if not legacy_scope and not allowed_doc_ids:
        return None
    # Expression form of the statute point lookup: the unified chunk corpus via
    # its statute identity attributes, newest applicable year first
    # (DESC NULLS LAST), exact-equality on instrument_name/section_ref.
    columns = (
        cast(UnifiedChunk.id, String),
        UnifiedChunk.instrument_name,
        UnifiedChunk.section_ref,
        UnifiedChunk.content,
        cast(UnifiedChunk.document_id, String),
        UnifiedChunk.instrument_year,
    )
    base = select(*columns).where(
        UnifiedChunk.instrument_name == act_name.strip(),
        UnifiedChunk.section_ref == section_ref.strip(),
    )
    if legacy_scope:
        query = base.where(UnifiedChunk.user_id == user_id)
    else:
        query = base.where(
            UnifiedChunk.tenant_id == tenant_id,
            UnifiedChunk.document_id == uuid_any("allowed_doc_ids", allowed_doc_ids or ()),
        )
    query = query.order_by(UnifiedChunk.instrument_year.desc().nulls_last()).limit(1)
    async with db_engine.connect() as conn:
        row = (await conn.execute(query)).fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "act_name": row[1],
            "section_ref": row[2],
            "title": f"{row[1]} {row[2]}",
            "body": row[3],
            "source_document_id": row[4],
            "jurisdiction": None,
            "year": row[5],
        }
