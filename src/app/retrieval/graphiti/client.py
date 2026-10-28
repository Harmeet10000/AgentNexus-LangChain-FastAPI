"""
Graphiti: async wrapper for the legal domain knowledge graph.

Design decisions:
- Graphiti creates its OWN async Neo4j driver internally. We pass the URI +
  credentials (not app.state.neo4j_driver) because graphiti-core does not
  accept an existing driver instance. The app.state.neo4j_driver remains
  available for raw Cypher queries.
- All public functions return domain types from schemas.py — no raw
  graphiti-core types leak outside this module.
- Pure async functions for setup and all CRUD operations.
- Helper functions remain private (_score_and_convert, _recency_score, etc.)
"""

from __future__ import annotations

import json
import traceback
from asyncio import TimeoutError as AsyncTimeoutError
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import NAMESPACE_URL, uuid5

from graphiti_core.errors import GraphitiError
from graphiti_core.nodes import EpisodeType
from neo4j.exceptions import Neo4jError

from app.observability.logging import logger

from .grouping import (
    GraphitiDocumentScope,
    build_graphiti_group_id,
    fanout_graphiti_group_ids,
)
from .schemas import GraphitiAvailability, GraphitiSearchOutcome, GraphitiSearchResult

_GRAPHITI_TRANSPORT_ERRORS = (GraphitiError, Neo4jError, AsyncTimeoutError, OSError)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from graphiti_core import Graphiti

    from .schemas import (
        ClauseEpisodeMetadata,
        LegalEdgeInput,
    )


def _scope_unavailable(*, requested: int) -> GraphitiSearchOutcome:
    """Represent an authorized allowlist with no resolvable graph groups."""
    return GraphitiSearchOutcome(
        availability=GraphitiAvailability.UNAVAILABLE,
        coverage={"requested": requested, "returned": 0, "reason": "scope_unavailable"},
    )


class BoundGraphitiService:
    """Bind the module's pure Graphiti operations to one process-owned client."""

    __slots__ = ("_graphiti",)

    def __init__(self, graphiti: Graphiti) -> None:
        self._graphiti = graphiti

    async def write_clause_episode(
        self,
        clause_text: str,
        metadata: ClauseEpisodeMetadata,
    ) -> str:
        return await write_clause_episode(self._graphiti, clause_text, metadata)

    async def write_relationship_edge(self, edge: LegalEdgeInput) -> str:
        return await write_relationship_edge(self._graphiti, edge)

    async def search_for_risk_context(
        self,
        query: str,
        user_id: str,
        *,
        doc_id: str | None = None,
        num_results: int = 10,
        allowed_doc_ids: tuple[str, ...] = (),
        tenant_id: str | None = None,
        contract_index: int | None = None,
        document_versions: Sequence[tuple[str, int]] = (),
        document_contracts: Sequence[tuple[str, int, Sequence[int | None], bool]] = (),
    ) -> GraphitiSearchOutcome:
        return await search_for_risk_context(
            self._graphiti,
            query,
            user_id,
            doc_id,
            num_results,
            allowed_doc_ids=allowed_doc_ids,
            tenant_id=tenant_id,
            contract_index=contract_index,
            document_versions=document_versions,
            document_contracts=document_contracts,
        )

    async def search_for_precedent_chains(
        self,
        query: str,
        user_id: str,
        *,
        jurisdiction: str = "India",
        num_results: int = 10,
        allowed_doc_ids: tuple[str, ...] = (),
        tenant_id: str | None = None,
        contract_index: int | None = None,
        document_versions: Sequence[tuple[str, int]] = (),
        document_contracts: Sequence[tuple[str, int, Sequence[int | None], bool]] = (),
    ) -> GraphitiSearchOutcome:
        return await search_for_precedent_chains(
            self._graphiti,
            query,
            user_id,
            jurisdiction,
            num_results,
            allowed_doc_ids=allowed_doc_ids,
            tenant_id=tenant_id,
            contract_index=contract_index,
            document_versions=document_versions,
            document_contracts=document_contracts,
        )

    async def get_obligation_chain(
        self,
        entity_name: str,
        user_id: str,
        *,
        doc_id: str | None = None,
        depth: int = 3,
        allowed_doc_ids: tuple[str, ...] = (),
        tenant_id: str | None = None,
        contract_index: int | None = None,
        document_versions: Sequence[tuple[str, int]] = (),
        document_contracts: Sequence[tuple[str, int, Sequence[int | None], bool]] = (),
    ) -> GraphitiSearchOutcome:
        return await get_obligation_chain(
            self._graphiti,
            entity_name,
            user_id,
            doc_id,
            depth,
            allowed_doc_ids=allowed_doc_ids,
            tenant_id=tenant_id,
            contract_index=contract_index,
            document_versions=document_versions,
            document_contracts=document_contracts,
        )


# Score weights for multi-objective memory retrieval (Section 18.6)
_W_SEMANTIC: float = 0.50
_W_RECENCY: float = 0.20
_W_TRUST: float = 0.20
_W_TASK_RELEVANCE: float = 0.10
# ---------------------------------------------------------------------------
# WRITE: clause episode
# ---------------------------------------------------------------------------


async def write_clause_episode(
    graphiti: Graphiti,
    clause_text: str,
    metadata: ClauseEpisodeMetadata,
) -> str:
    """Write a single clause as a Graphiti episode.

    Args:
        graphiti: Graphiti instance.
        clause_text: The clause text to store.
        metadata: Clause metadata (doc_id, clause_id, etc.).

    Returns:
        Episode UUID assigned by Graphiti.

    Raises:
        Exception: If write fails.
    """
    source_description = metadata.model_dump_json()
    episode_name = (
        f"clause:{metadata.tenant_id}:{metadata.doc_id}:v{metadata.document_version}:"
        f"c{metadata.contract_index}:{metadata.clause_id}"
    )
    episode_uuid = str(uuid5(NAMESPACE_URL, f"agentnexus:{episode_name}"))

    logger.bind(
        service="graphiti",
        doc_id=metadata.doc_id,
        clause_id=metadata.clause_id,
    ).info("Writing clause episode to Graphiti")

    try:
        await graphiti.add_episode(
            name=episode_name,
            episode_body=clause_text,
            source=EpisodeType.text,
            source_description=source_description,
            reference_time=datetime.now(tz=UTC),
            group_id=build_graphiti_group_id(
                tenant_id=metadata.tenant_id,
                document_id=metadata.doc_id,
                document_version=metadata.document_version,
                contract_index=metadata.contract_index,
            ),
            uuid=episode_uuid,
        )
    except _GRAPHITI_TRANSPORT_ERRORS as exc:
        exc.add_note("operation=write_clause_episode, service=graphiti")
        exc.add_note(traceback.format_exc())
        logger.bind(
            service="graphiti",
            operation="write_clause_episode",
            error_type=type(exc).__name__,
        ).exception("Failed to write clause episode")
        raise
    return episode_uuid


# ---------------------------------------------------------------------------
# WRITE: structured relationship edge
# ---------------------------------------------------------------------------


async def write_relationship_edge(
    graphiti: Graphiti,
    edge: LegalEdgeInput,
) -> str:
    """Write a legal relationship as a structured Graphiti episode.

    Args:
        graphiti: Graphiti instance.
        edge: Relationship edge with entities and relationship type.

    Returns:
        Episode UUID assigned by Graphiti.

    Raises:
        Exception: If write fails.
    """
    episode_name = (
        f"edge:{edge.tenant_id}:{edge.doc_id}:v{edge.document_version}:c{edge.contract_index}:"
        f"{edge.clause_id}:{edge.from_entity}:{edge.relationship}:{edge.to_entity}"
    )
    episode_uuid = str(uuid5(NAMESPACE_URL, f"agentnexus:{episode_name}"))
    source_description = json.dumps(
        {
            "type": "relationship_edge",
            "doc_id": edge.doc_id,
            "clause_id": edge.clause_id,
            "user_id": edge.user_id,
            "thread_id": edge.thread_id,
            "confidence": edge.confidence,
        }
    )
    logger.bind(
        service="graphiti",
        from_entity=edge.from_entity,
        relationship=edge.relationship,
        to_entity=edge.to_entity,
    ).info("Writing relationship edge to Graphiti")

    try:
        await graphiti.add_episode(
            name=episode_name,
            episode_body=edge.to_episode_body(),
            source=EpisodeType.text,
            source_description=source_description,
            reference_time=datetime.now(tz=UTC),
            group_id=build_graphiti_group_id(
                tenant_id=edge.tenant_id,
                document_id=edge.doc_id,
                document_version=edge.document_version,
                contract_index=edge.contract_index,
            ),
            uuid=episode_uuid,
        )
    except _GRAPHITI_TRANSPORT_ERRORS as exc:
        exc.add_note("operation=write_relationship_edge, service=graphiti")
        exc.add_note(traceback.format_exc())
        logger.bind(
            service="graphiti",
            operation="write_relationship_edge",
            error_type=type(exc).__name__,
        ).exception("Failed to write relationship edge")
        raise
    return episode_uuid


# ---------------------------------------------------------------------------
# READ: risk analysis context
# ---------------------------------------------------------------------------


async def search_for_risk_context(
    graphiti: Graphiti,
    query: str,
    user_id: str,
    doc_id: str | None = None,
    num_results: int = 10,
    *,
    allowed_doc_ids: tuple[str, ...] = (),
    tenant_id: str | None = None,
    contract_index: int | None = None,
    document_versions: Sequence[tuple[str, int]] = (),
    document_contracts: Sequence[tuple[str, int, Sequence[int | None], bool]] = (),
) -> GraphitiSearchOutcome:
    """Search the knowledge graph for risk-relevant context.

    Searches across the server-authorized canonical tenant/document/version/
    contract partitions for clause-level context and precedent patterns.

    Args:
        graphiti: Graphiti instance.
        query: Search query string.
        user_id: User ID for scoping search.
        doc_id: Optional document ID for current-doc context.
        num_results: Number of results to return.

    Returns:
        List of scored GraphitiSearchResult items.
    """
    # Explicit document scope is authoritative. An empty allowlist is an
    # authorization boundary, not a request to widen to historical groups.
    del user_id, doc_id
    if not allowed_doc_ids:
        return GraphitiSearchOutcome()
    group_ids = _scoped_group_ids(
        allowed_doc_ids,
        tenant_id=tenant_id,
        contract_index=contract_index,
        document_versions=document_versions,
        document_contracts=document_contracts,
    )
    # An allowlist without a live tenant is not an authorization scope.  Do
    # not pass an empty ``group_ids`` list to Graphiti: providers may interpret
    # that as "all groups" rather than "no groups".
    if not group_ids:
        return _scope_unavailable(requested=num_results)

    logger.bind(
        service="graphiti",
        query=query[:80],
        group_ids=group_ids,
    ).info("Searching Graphiti for risk context")

    try:
        raw_results = await graphiti.search(
            query=query,
            group_ids=group_ids,
            num_results=num_results,
        )
    except _GRAPHITI_TRANSPORT_ERRORS as exc:
        exc.add_note(f"query={query[:80]}, task=risk_analysis")
        exc.add_note(traceback.format_exc())
        logger.bind(service="graphiti").exception("Graphiti risk context search failed")
        return GraphitiSearchOutcome(
            availability=GraphitiAvailability.UNAVAILABLE,
            coverage={"requested": num_results, "returned": 0, "error": type(exc).__name__},
        )
    else:
        results = [
            _score_and_convert(r, query=query, task="risk_analysis") for r in (raw_results or [])
        ]
        return GraphitiSearchOutcome(
            results,
            coverage={"requested": num_results, "returned": len(results), "group_ids": group_ids},
        )


# ---------------------------------------------------------------------------
# READ: precedent chains
# ---------------------------------------------------------------------------


async def search_for_precedent_chains(
    graphiti: Graphiti,
    query: str,
    user_id: str,
    jurisdiction: str = "India",
    num_results: int = 10,
    *,
    allowed_doc_ids: tuple[str, ...] = (),
    tenant_id: str | None = None,
    contract_index: int | None = None,
    document_versions: Sequence[tuple[str, int]] = (),
    document_contracts: Sequence[tuple[str, int, Sequence[int | None], bool]] = (),
) -> GraphitiSearchOutcome:
    """Search the knowledge graph for compliance precedents.

    Searches user's historical approved reports — the highest-trust
    episodic memory — to find how similar clauses were handled before.

    Args:
        graphiti: Graphiti instance.
        query: Search query string.
        user_id: User ID to search within.
        jurisdiction: Filter results by jurisdiction.
        num_results: Number of results to return.

    Returns:
        List of scored precedent results.
    """
    del user_id
    if not allowed_doc_ids:
        return GraphitiSearchOutcome()

    logger.bind(
        service="graphiti",
        query=query[:80],
        jurisdiction=jurisdiction,
    ).info("Searching Graphiti for precedents")

    try:
        group_ids = _scoped_group_ids(
            allowed_doc_ids,
            tenant_id=tenant_id,
            contract_index=contract_index,
            document_versions=document_versions,
            document_contracts=document_contracts,
        )
        if not group_ids:
            return _scope_unavailable(requested=num_results)
        raw_results = await graphiti.search(
            query=query,
            group_ids=group_ids,
            num_results=num_results * 2,
        )

        filtered = [r for r in (raw_results or []) if _has_jurisdiction(r, jurisdiction)][
            :num_results
        ]
    except _GRAPHITI_TRANSPORT_ERRORS as exc:
        exc.add_note(f"query={query[:80]}, task=precedent_chains")
        exc.add_note(traceback.format_exc())
        logger.bind(service="graphiti").exception("Graphiti precedent search failed")
        return GraphitiSearchOutcome(
            availability=GraphitiAvailability.UNAVAILABLE,
            coverage={"requested": num_results, "returned": 0, "error": type(exc).__name__},
        )
    else:
        results = [_score_and_convert(r, query=query, task="compliance") for r in filtered]
        return GraphitiSearchOutcome(
            results,
            coverage={"requested": num_results, "returned": len(results)},
        )


# ---------------------------------------------------------------------------
# READ: obligation chain
# ---------------------------------------------------------------------------


async def get_obligation_chain(
    graphiti: Graphiti,
    entity_name: str,
    user_id: str,
    doc_id: str | None = None,
    depth: int = 3,
    *,
    allowed_doc_ids: tuple[str, ...] = (),
    tenant_id: str | None = None,
    contract_index: int | None = None,
    document_versions: Sequence[tuple[str, int]] = (),
    document_contracts: Sequence[tuple[str, int, Sequence[int | None], bool]] = (),
) -> GraphitiSearchOutcome:
    """Forward-chain obligations from a named entity.

    Example: given "Acme Corp", returns:
      Acme Corp → INDEMNIFIES → GlobalTech Ltd
      Acme Corp → OBLIGES → Event: delivery by 2025-12-31
      Event: delivery → TRIGGERS → Penalty clause

    Args:
        graphiti: Graphiti instance.
        entity_name: Named entity to find obligations for.
        user_id: User ID for scoping search.
        doc_id: Optional document ID to scope further.
        depth: Hop count in the graph traversal.

    Returns:
        List of obligation chain results.
    """
    del user_id, doc_id
    if not allowed_doc_ids:
        return GraphitiSearchOutcome()
    group_ids = _scoped_group_ids(
        allowed_doc_ids,
        tenant_id=tenant_id,
        contract_index=contract_index,
        document_versions=document_versions,
        document_contracts=document_contracts,
    )
    if not group_ids:
        return _scope_unavailable(requested=depth * 5)

    obligation_query = f"obligations responsibilities duties of {entity_name}"
    logger.bind(
        service="graphiti",
        entity=entity_name,
        depth=depth,
    ).info("Retrieving Graphiti obligation chain")

    try:
        raw_results = await graphiti.search(
            query=obligation_query,
            group_ids=group_ids,
            num_results=depth * 5,
        )
    except _GRAPHITI_TRANSPORT_ERRORS as exc:
        exc.add_note(f"query={obligation_query[:80]}, task=obligation_chain")
        exc.add_note(traceback.format_exc())
        logger.bind(service="graphiti").exception("Graphiti obligation chain search failed")
        return GraphitiSearchOutcome(
            availability=GraphitiAvailability.UNAVAILABLE,
            coverage={"requested": depth * 5, "returned": 0, "error": type(exc).__name__},
        )
    else:
        results = [
            _score_and_convert(r, query=obligation_query, task="obligation_chain")
            for r in (raw_results or [])
        ]
        return GraphitiSearchOutcome(
            results,
            coverage={"requested": depth * 5, "returned": len(results), "group_ids": group_ids},
        )


def _scoped_group_ids(
    document_ids: tuple[str, ...],
    *,
    tenant_id: str | None,
    contract_index: int | None,
    document_versions: Sequence[tuple[str, int]],
    document_contracts: Sequence[tuple[str, int, Sequence[int | None], bool]] = (),
) -> list[str]:
    """Build exact authorized tenant/document/version/contract partitions.

    The legacy ``contract_index`` argument is retained for callers that have
    already narrowed to one contract.  Without it, modern callers must supply
    ``document_contracts``; an absent contract scope returns no groups rather
    than querying the shared ``contract_none`` partition.
    """
    if not tenant_id:
        return []
    versions = {str(item[0]): int(item[1]) for item in document_versions}
    contracts = {
        document_id: (int(version), tuple(indexes), legacy_null)
        for document_id, version, indexes, legacy_null in document_contracts
    }
    scopes: list[GraphitiDocumentScope] = []
    for document_id in document_ids:
        scoped = contracts.get(document_id)
        if scoped is not None:
            version, scoped_indexes, scoped_legacy_null = scoped
        else:
            version = versions.get(document_id)
            # A missing version is not a safe request for the legacy v1
            # partition: fail closed rather than widening the graph scope.
            if version is None:
                continue
            scoped_indexes, scoped_legacy_null = (), False
        if contract_index is not None:
            indexes: tuple[int | None, ...] = (contract_index,)
            legacy_null = False
        else:
            indexes, legacy_null = scoped_indexes, scoped_legacy_null
        scopes.append(
            GraphitiDocumentScope(
                tenant_id=tenant_id,
                document_id=document_id,
                document_version=version,
                contract_indexes=indexes,
                legacy_null_contract=legacy_null,
            )
        )
    return fanout_graphiti_group_ids(scopes)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _score_and_convert(
    raw: Any,
    query: str,
    task: str,
) -> GraphitiSearchResult:
    """Convert a raw graphiti-core result to GraphitiSearchResult.

    Multi-objective score (Section 18.6):
      score = w1 * semantic_similarity + w2 * recency + w3 * trust_score + w4 * task_relevance
    """
    # Extract raw scores
    semantic = float(getattr(raw, "score", 0.5))

    # Recency: episodes from the last 30 days score 1.0, decaying to 0.0 at 365 days
    created_at: datetime | None = getattr(raw, "created_at", None)
    recency = _recency_score(created_at)

    # Trust: human_reviewed flag in source_description bumps score
    source_desc: str = getattr(raw, "source_description", "") or ""
    trust = _trust_score(source_desc)

    # Task relevance: simple keyword presence
    content: str = getattr(raw, "content", "") or getattr(raw, "episode_body", "") or ""
    task_rel = _task_relevance(content, query)

    final_score = (
        _W_SEMANTIC * semantic
        + _W_RECENCY * recency
        + _W_TRUST * trust
        + _W_TASK_RELEVANCE * task_rel
    )

    return GraphitiSearchResult(
        uuid=str(getattr(raw, "uuid", "")),
        name=getattr(raw, "name", ""),
        content=content,
        source_description=source_desc,
        relevance_score=min(final_score, 1.0),
        group_id=getattr(raw, "group_id", None),
        created_at=created_at,
        metadata_raw={"task": task, "raw_score": semantic},
    )


def _recency_score(created_at: datetime | None) -> float:
    """Calculate recency score based on episode age."""
    if created_at is None:
        return 0.5
    age_days = (datetime.now(tz=UTC) - created_at).days
    return max(0.0, 1.0 - age_days / 365.0)


def _trust_score(source_description: str) -> float:
    """Extract trust score from metadata."""
    if not source_description:
        return 0.5
    try:
        meta = json.loads(source_description)
        if meta.get("human_reviewed") or meta.get("human_approved"):
            return 1.0
        return float(meta.get("trust_score", 0.5))
    except (json.JSONDecodeError, TypeError):
        return 0.5


def _task_relevance(content: str, query: str) -> float:
    """Compute task relevance as query-content word overlap."""
    if not content or not query:
        return 0.0
    query_words = set(query.lower().split())
    content_words = set(content.lower().split())
    overlap = len(query_words & content_words)
    return min(1.0, overlap / max(len(query_words), 1))


def _has_jurisdiction(raw: Any, jurisdiction: str) -> bool:
    """Check if episode matches the given jurisdiction."""
    source_desc: str = getattr(raw, "source_description", "") or ""
    if not source_desc:
        return True
    try:
        meta = json.loads(source_desc)
        stored_jurisdiction = meta.get("jurisdiction", "")
        return not stored_jurisdiction or jurisdiction.lower() in stored_jurisdiction.lower()
    except (json.JSONDecodeError, TypeError):
        return True
