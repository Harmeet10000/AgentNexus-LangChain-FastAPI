"""Graphiti write and verification helpers for legal chunks."""

from __future__ import annotations

import re
from datetime import UTC, datetime

from graphiti_core.nodes import EpisodeType
from pydantic import BaseModel, ConfigDict

from app.utils import logger


class GraphitiVerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str
    episode_id: str | None
    verified: bool


async def write_and_verify_chunk(
    *,
    graphiti: object | None,
    user_id: str,
    document_id: str,
    chunk_id: str,
    clause_type: str | None,
    preamble: str,
    content: str,
) -> GraphitiVerificationResult:
    if graphiti is None:
        return GraphitiVerificationResult(chunk_id=chunk_id, episode_id=None, verified=False)
    body = f"{preamble}\n\n{content}\n\nREFERENCES_CLAUSE postgres_chunk_id={chunk_id}"
    source_description = (
        "{"
        f'"user_id":"{user_id}",'
        f'"doc_id":"{document_id}",'
        f'"postgres_chunk_id":"{chunk_id}",'
        f'"clause_type":"{clause_type or "other"}"'
        "}"
    )
    try:
        result = await graphiti.add_episode(  # type: ignore[attr-defined]
            name=f"chunk:{document_id}:{chunk_id}",
            episode_body=body,
            source=EpisodeType.text,
            source_description=source_description,
            reference_time=datetime.now(tz=UTC),
            group_id=document_id,
        )
        episode_id = str(getattr(result, "uuid", "")) or None
    except Exception as exc:  # noqa: BLE001
        logger.bind(document_id=document_id, chunk_id=chunk_id).warning(
            "graphiti_chunk_write_failed",
            error=str(exc),
        )
        return GraphitiVerificationResult(chunk_id=chunk_id, episode_id=None, verified=False)

    try:
        raw_results = await graphiti.search(  # type: ignore[attr-defined]
            query=chunk_id,
            group_ids=[user_id, document_id],
            num_results=10,
        )
    except Exception as exc:  # noqa: BLE001
        logger.bind(document_id=document_id, chunk_id=chunk_id).warning(
            "graphiti_chunk_verify_failed",
            error=str(exc),
        )
        return GraphitiVerificationResult(chunk_id=chunk_id, episode_id=episode_id, verified=False)

    verified = any(
        chunk_id in _extract_postgres_chunk_ids(_extract_search_blob(item))
        for item in raw_results or []
    )
    return GraphitiVerificationResult(chunk_id=chunk_id, episode_id=episode_id, verified=verified)


def _extract_search_blob(item: object) -> str:
    source_description = getattr(item, "source_description", "") or ""
    content = getattr(item, "content", "") or getattr(item, "episode_body", "") or ""
    return f"{source_description}\n{content}"


def _extract_postgres_chunk_ids(value: str) -> list[str]:
    return re.findall(
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
        value,
    )
