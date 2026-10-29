"""Graphiti write and verification helpers for legal chunks."""

from __future__ import annotations

import re
import traceback
from asyncio import TimeoutError as AsyncTimeoutError
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from graphiti_core.errors import GraphitiError
from graphiti_core.nodes import EpisodeType
from neo4j.exceptions import Neo4jError
from pydantic import BaseModel, ConfigDict, Field

from app.observability.logging import logger
from app.retrieval.graphiti.grouping import build_graphiti_group_id

_GRAPHITI_TRANSPORT_ERRORS = (GraphitiError, Neo4jError, AsyncTimeoutError, OSError)

if TYPE_CHECKING:
    from graphiti_core.edges import EntityEdge
    from graphiti_core.graphiti import AddEpisodeResults, Graphiti


class GraphitiVerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str
    episode_id: str | None
    verified: bool
    availability: str = "available"
    coverage: dict[str, object] = Field(default_factory=dict)


async def write_and_verify_chunk(
    *,
    graphiti: Graphiti | None,
    user_id: str,
    document_id: str,
    chunk_id: str,
    clause_type: str | None,
    preamble: str,
    content: str,
    tenant_id: str = "",
    contract_index: int | None = None,
    document_version: int = 1,
) -> GraphitiVerificationResult:
    if graphiti is None:
        return GraphitiVerificationResult(
            chunk_id=chunk_id,
            episode_id=None,
            verified=False,
            availability="unavailable",
            coverage={"requested": 1, "returned": 0},
        )
    body = f"{preamble}\n\n{content}\n\nREFERENCES_CLAUSE postgres_chunk_id={chunk_id}"
    source_description = (
        "{"
        f'"user_id":"{user_id}",'
        f'"doc_id":"{document_id}",'
        f'"tenant_id":"{tenant_id}",'
        f'"document_version":{document_version},'
        f'"contract_index":{contract_index if contract_index is not None else "null"},'
        f'"postgres_chunk_id":"{chunk_id}",'
        f'"clause_type":"{clause_type or "other"}"'
        "}"
    )
    try:
        result: AddEpisodeResults = await graphiti.add_episode(
            name=f"chunk:{document_id}:{chunk_id}",
            episode_body=body,
            source=EpisodeType.text,
            source_description=source_description,
            reference_time=datetime.now(tz=UTC),
            group_id=build_graphiti_group_id(
                tenant_id=tenant_id,
                document_id=document_id,
                document_version=document_version,
                contract_index=contract_index,
            ),
        )
        episode_id = str(getattr(result, "uuid", "")) or None
    except _GRAPHITI_TRANSPORT_ERRORS as exc:
        exc.add_note(f"document_id={document_id}, chunk_id={chunk_id}")
        exc.add_note(traceback.format_exc())
        logger.bind(
            document_id=document_id,
            chunk_id=chunk_id,
            error_type=type(exc).__name__,
        ).exception(
            "graphiti_chunk_write_failed",
        )
        return GraphitiVerificationResult(
            chunk_id=chunk_id,
            episode_id=None,
            verified=False,
            availability="unavailable",
            coverage={"requested": 1, "returned": 0, "error": type(exc).__name__},
        )

    try:
        raw_results: list[EntityEdge] = await graphiti.search(
            query=chunk_id,
            group_ids=[
                build_graphiti_group_id(
                    tenant_id=tenant_id,
                    document_id=document_id,
                    document_version=document_version,
                    contract_index=contract_index,
                )
            ],
            num_results=10,
        )
    except _GRAPHITI_TRANSPORT_ERRORS as exc:
        exc.add_note(f"document_id={document_id}, chunk_id={chunk_id}, operation=search")
        exc.add_note(traceback.format_exc())
        logger.bind(
            document_id=document_id,
            chunk_id=chunk_id,
            error_type=type(exc).__name__,
        ).exception(
            "graphiti_chunk_verify_failed",
        )
        return GraphitiVerificationResult(
            chunk_id=chunk_id,
            episode_id=episode_id,
            verified=False,
            availability="unavailable",
            coverage={"requested": 1, "returned": 0, "error": type(exc).__name__},
        )

    verified: bool = any(
        chunk_id in _extract_postgres_chunk_ids(_extract_search_blob(item))
        for item in raw_results or []
    )
    return GraphitiVerificationResult(
        chunk_id=chunk_id,
        episode_id=episode_id,
        verified=verified,
        coverage={"requested": 1, "returned": 1 if verified else 0},
    )


def _extract_search_blob(item: object) -> str:
    source_description = getattr(item, "source_description", "") or ""
    content = getattr(item, "content", "") or getattr(item, "episode_body", "") or ""
    return f"{source_description}\n{content}"


def _extract_postgres_chunk_ids(value: str) -> list[str]:
    return re.findall(
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
        value,
    )
