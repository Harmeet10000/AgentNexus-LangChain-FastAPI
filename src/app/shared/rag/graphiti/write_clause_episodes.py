"""
Tool: write_clause_episodes

Used by relationship_mapping node to persist clause episodes + edges to Graphiti.

Writes in two phases:
  Phase 1: Parallel clause episode writes (one per segment, N concurrent)
  Phase 2: Sequential relationship edge writes (graph edges depend on episodes existing)

Concurrency strategy (Section async rules):
  - Phase 1 uses asyncio.gather() — bounded, known task set (clause count per doc)
  - A semaphore caps concurrent Graphiti writes at _MAX_CONCURRENT_WRITES
    to avoid overwhelming Neo4j's connection pool

Idempotency:
  - Idempotency key per clause: hash(doc_id + clause_id + user_id)
  - Already-written episodes are skipped without error
  - This makes the relationship_mapping node safely retryable

These are NOT LangChain @tool decorated — they're called directly from
the relationship_mapping node function (not via create_react_agent).
They're structured as an async service function, not a tool, because:
  1. The LLM should never decide WHEN to write — the node always writes
  2. No tool selection ambiguity needed
  3. Batch write pattern doesn't map to single-call @tool interface
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from app.shared.langchain_layer.agents.tools.idempotency import IdempotencyGuard, ToolResult
from app.utils import logger

from .schemas import ClauseEpisodeMetadata, LegalEdgeInput

if TYPE_CHECKING:
    from app.shared.langgraph_layer.agent_saul.state import ClauseSegment, LegalRelationship

_MAX_CONCURRENT_WRITES: int = 5


class ClauseWriteResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    clause_id: str
    episode_uuid: str
    success: bool
    error: str | None = None


class RelationshipWriteResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    edge_id: str
    episode_uuid: str
    success: bool
    error: str | None = None


async def write_clause_episodes_to_graphiti(
    segments: list[ClauseSegment],
    relationships: list[LegalRelationship],
    doc_id: str,
    user_id: str,
    thread_id: str,
    jurisdiction: str,
    document_type: str,
    human_reviewed: bool,
    graphiti_service: GraphitiService,
    idempotency: IdempotencyGuard,
) -> tuple[list[ClauseWriteResult], list[RelationshipWriteResult]]:
    """Write clause episodes + relationship edges to Graphiti.

    Returns (clause_results, relationship_results).
    Partial failures are recorded per-item — caller decides on retry.
    """
    log = logger.bind(
        doc_id=doc_id,
        clause_count=len(segments),
        relationship_count=len(relationships),
    )

    # Phase 1: Parallel clause episode writes (bounded by semaphore)
    sem = asyncio.Semaphore(_MAX_CONCURRENT_WRITES)
    clause_tasks = [
        _write_single_clause_episode(
            segment=seg,
            doc_id=doc_id,
            user_id=user_id,
            thread_id=thread_id,
            jurisdiction=jurisdiction,
            document_type=document_type,
            human_reviewed=human_reviewed,
            graphiti_service=graphiti_service,
            idempotency=idempotency,
            sem=sem,
        )
        for seg in segments
    ]
    clause_results: list[ClauseWriteResult] = list(
        await asyncio.gather(*clause_tasks, return_exceptions=False)
    )
    successful_clauses = sum(1 for r in clause_results if r.success)
    log.info("clause_episodes_written", successful=successful_clauses, total=len(segments))

    # Phase 2: Relationship edges (sequential — edges reference episode entities)
    rel_results: list[RelationshipWriteResult] = []
    for rel in relationships:
        result = await _write_single_relationship_edge(
            relationship=rel,
            doc_id=doc_id,
            user_id=user_id,
            thread_id=thread_id,
            graphiti_service=graphiti_service,
            idempotency=idempotency,
        )
        rel_results.append(result)

    successful_rels = sum(1 for r in rel_results if r.success)
    log.info(
        "relationship_edges_written",
        successful=successful_rels,
        total=len(relationships),
    )

    return clause_results, rel_results


async def _write_single_clause_episode(
    segment: ClauseSegment,
    doc_id: str,
    user_id: str,
    thread_id: str,
    jurisdiction: str,
    document_type: str,
    human_reviewed: bool,
    graphiti_service: GraphitiService,
    idempotency: IdempotencyGuard,
    sem: asyncio.Semaphore,
) -> ClauseWriteResult:
    idem_key = IdempotencyGuard.make_key(
        step_id=f"clause_episode:{segment.clause_id}",
        input_data={"doc_id": doc_id, "clause_id": segment.clause_id},
        user_id=user_id,
    )

    cached = await idempotency.get(idem_key)
    if cached is not None and cached.success:
        return ClauseWriteResult(
            clause_id=segment.clause_id,
            episode_uuid=cached.data.get("episode_uuid", ""),
            success=True,
        )

    async with sem:
        try:
            metadata = ClauseEpisodeMetadata(
                doc_id=doc_id,
                clause_id=segment.clause_id,
                clause_type=segment.clause_type.value,
                jurisdiction=jurisdiction,
                document_type=document_type,
                user_id=user_id,
                thread_id=thread_id,
                human_reviewed=human_reviewed,
                trust_score=0.9 if human_reviewed else 0.5,
            )
            uuid = await graphiti_service.write_clause_episode(
                clause_text=segment.text,
                metadata=metadata,
            )

            await idempotency.set(
                key=idem_key,
                result=ToolResult.ok(
                    data={"episode_uuid": uuid, "clause_id": segment.clause_id},
                ),
                tool_name="write_clause_episode",
                user_id=user_id,
                thread_id=thread_id,
                step_id=f"clause_episode:{segment.clause_id}",
            )
            return ClauseWriteResult(
                clause_id=segment.clause_id,
                episode_uuid=uuid,
                success=True,
            )
        except Exception as exc:
            logger.bind(
                clause_id=segment.clause_id,
                error=str(exc),
            ).exception("clause_episode_write_failed")
            return ClauseWriteResult(
                clause_id=segment.clause_id,
                episode_uuid="",
                success=False,
                error=str(exc),
            )


async def _write_single_relationship_edge(
    relationship: LegalRelationship,
    doc_id: str,
    user_id: str,
    thread_id: str,
    graphiti_service: GraphitiService,
    idempotency: IdempotencyGuard,
) -> RelationshipWriteResult:
    idem_key = IdempotencyGuard.make_key(
        step_id=f"rel_edge:{relationship.edge_id}",
        input_data={"doc_id": doc_id, "edge_id": relationship.edge_id},
        user_id=user_id,
    )

    cached = await idempotency.get(idem_key)
    if cached is not None and cached.success:
        return RelationshipWriteResult(
            edge_id=relationship.edge_id,
            episode_uuid=cached.data.get("episode_uuid", ""),
            success=True,
        )

    try:
        edge_input = LegalEdgeInput(
            from_entity=relationship.from_node,
            relationship=relationship.relationship.value,
            to_entity=relationship.to_node,
            clause_id=relationship.clause_id,
            doc_id=doc_id,
            user_id=user_id,
            thread_id=thread_id,
            citation_source=relationship.citation.source,
            confidence=relationship.citation.confidence,
        )
        uuid = await graphiti_service.write_relationship_edge(edge_input)

        await idempotency.set(
            key=idem_key,
            result=ToolResult.ok(
                data={"episode_uuid": uuid, "edge_id": relationship.edge_id},
            ),
            tool_name="write_relationship_edge",
            user_id=user_id,
            thread_id=thread_id,
            step_id=f"rel_edge:{relationship.edge_id}",
        )
        return RelationshipWriteResult(
            edge_id=relationship.edge_id,
            episode_uuid=uuid,
            success=True,
        )
    except Exception as exc:
        logger.bind(
            edge_id=relationship.edge_id,
            error=str(exc),
        ).exception("relationship_edge_write_failed")
        return RelationshipWriteResult(
            edge_id=relationship.edge_id,
            episode_uuid="",
            success=False,
            error=str(exc),
        )
