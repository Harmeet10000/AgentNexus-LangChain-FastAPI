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
import re
import traceback
from asyncio import TimeoutError as AsyncTimeoutError
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import NAMESPACE_URL, uuid5

from graphiti_core.errors import GraphitiError
from graphiti_core.nodes import EpisodeType
from neo4j.exceptions import Neo4jError

# The closed relationship vocabulary is owned by the pass-2 producer. Reads
# parse persisted edge episodes against the same enum so an episode whose
# relationship is not in the vocabulary fails closed (dropped, never typed).
from app.features.agent_saul.domain.state import RelationshipType
from app.observability.logging import logger
from app.retrieval.edge_expansion import LegalEdgeRecord

from .grouping import (
    GraphitiDocumentScope,
    build_graphiti_group_id,
    fanout_graphiti_group_ids,
)
from .schemas import GraphitiAvailability, GraphitiSearchOutcome, GraphitiSearchResult

_GRAPHITI_TRANSPORT_ERRORS = (GraphitiError, Neo4jError, AsyncTimeoutError, OSError)
# Trailing "Grounding: chars a-b." written by LegalEdgeInput.to_episode_body().
_EDGE_GROUNDING_PATTERN = re.compile(r"Grounding: chars (\d+)-(\d+)\.$")

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

    async def fetch_relationship_edges(
        self,
        query: str,
        *,
        allowed_doc_ids: tuple[str, ...] = (),
        tenant_id: str | None = None,
        contract_index: int | None = None,
        document_versions: Sequence[tuple[str, int, str | None]] = (),
        document_contracts: Sequence[tuple[str, int, Sequence[int | None], bool]] = (),
        num_results: int = 32,
    ) -> tuple[LegalEdgeRecord, ...]:
        return await fetch_relationship_edges(
            self._graphiti,
            query,
            allowed_doc_ids=allowed_doc_ids,
            tenant_id=tenant_id,
            contract_index=contract_index,
            document_versions=document_versions,
            document_contracts=document_contracts,
            num_results=num_results,
        )

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
            # Structured triple: the reader never re-parses prose — entity
            # names containing periods/abbreviations broke sentence splitting.
            "from_entity": edge.from_entity,
            "relationship": edge.relationship,
            "to_entity": edge.to_entity,
            "contract_index": edge.contract_index,
            "document_version": edge.document_version,
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
    document_versions: Sequence[tuple[str, int]] | Sequence[tuple[str, int, str | None]],
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
# READ: typed relationship edges (first retrieval consumer)
# ---------------------------------------------------------------------------


async def fetch_relationship_edges(
    graphiti: Graphiti,
    query: str,
    *,
    allowed_doc_ids: tuple[str, ...] = (),
    tenant_id: str | None = None,
    contract_index: int | None = None,
    document_versions: Sequence[tuple[str, int, str | None]] = (),
    document_contracts: Sequence[tuple[str, int, Sequence[int | None], bool]] = (),
    num_results: int = 32,
) -> tuple[LegalEdgeRecord, ...]:
    """Read typed relationship edges scoped to the authorized graph partitions.

    Same authorization scaffold as ``search_for_risk_context``: an empty
    allowlist is a fail-closed boundary, a scope without resolvable groups
    reads nothing, and every returned record was persisted by the pass-2
    producer (``source_description`` ``type == "relationship_edge"``).  A
    transport failure degrades to an empty tuple — expansion is additive and
    must never turn a healthy ask into a failed one.
    """
    if not allowed_doc_ids:
        return ()
    group_ids = _scoped_group_ids(
        allowed_doc_ids,
        tenant_id=tenant_id,
        contract_index=contract_index,
        document_versions=document_versions,
        document_contracts=document_contracts,
    )
    if not group_ids:
        return ()

    logger.bind(
        service="graphiti",
        query=query[:80],
        group_ids=group_ids,
    ).info("Fetching typed relationship edges")

    try:
        raw_results = await graphiti.search(
            query=query,
            group_ids=group_ids,
            num_results=num_results,
        )
    except _GRAPHITI_TRANSPORT_ERRORS as exc:
        exc.add_note(f"query={query[:80]}, task=relationship_edges")
        exc.add_note(traceback.format_exc())
        logger.bind(service="graphiti").exception("Graphiti relationship edge fetch failed")
        return ()
    edges = [
        record
        for raw in (raw_results or [])
        if (record := _parse_relationship_edge(raw)) is not None
    ]
    return tuple(edges)


def _parse_relationship_edge(raw: Any) -> LegalEdgeRecord | None:
    """Convert one raw episode into a ``LegalEdgeRecord``, or drop it.

    Only episodes the producer wrote (``source_description`` typed
    ``relationship_edge``) qualify.  Identity fields come from the
    server-written source description, which also carries the structured
    from/relationship/to triple; the prose body is parsed only as a fallback
    for rows written before the structured fields existed.  An untypeable
    edge is dropped, never guessed.
    """
    source_desc: str = getattr(raw, "source_description", "") or ""
    try:
        meta = json.loads(source_desc)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(meta, dict) or meta.get("type") != "relationship_edge":
        return None
    body: str = getattr(raw, "content", "") or getattr(raw, "episode_body", "") or ""
    structured = _structured_edge_triple(meta)
    if structured is not None:
        from_entity, relationship, to_entity = structured
    else:
        triple = _parse_edge_body_triple(body)
        if triple is None:
            return None
        from_entity, relationship, to_entity = triple
    grounding = _EDGE_GROUNDING_PATTERN.search(body)
    try:
        confidence = float(meta.get("confidence"))
    except (TypeError, ValueError):
        confidence = 0.5
    contract_index = meta.get("contract_index")
    document_version = meta.get("document_version")
    return LegalEdgeRecord(
        edge_id=str(getattr(raw, "uuid", "")),
        document_id=str(meta.get("doc_id") or ""),
        from_entity=from_entity,
        relationship=relationship,
        to_entity=to_entity,
        clause_id=str(meta.get("clause_id") or ""),
        char_start=int(grounding.group(1)) if grounding else None,
        char_end=int(grounding.group(2)) if grounding else None,
        confidence=max(0.0, min(confidence, 1.0)),
        group_id=getattr(raw, "group_id", None),
        contract_index=int(contract_index) if isinstance(contract_index, int) else None,
        document_version=(int(document_version) if isinstance(document_version, int) else None),
    )


def _structured_edge_triple(meta: dict[str, Any]) -> tuple[str, str, str] | None:
    """Read the triple from structured source-description fields.

    Present only on rows written after the structured fields existed; every
    part must be a non-empty string and the relationship must be in the closed
    vocabulary, else the structured path declines (and the caller falls back
    to the prose parse).
    """
    parts = (meta.get("from_entity"), meta.get("relationship"), meta.get("to_entity"))
    if not all(isinstance(part, str) and part for part in parts):
        return None
    from_entity, relationship, to_entity = (str(part) for part in parts)
    if relationship not in {member.value for member in RelationshipType}:
        return None
    return from_entity, relationship, to_entity


def _parse_edge_body_triple(body: str) -> tuple[str, str, str] | None:
    """Recover ``'{from} {relationship} {to}.'`` against the closed vocabulary.

    Legacy fallback for rows written before the structured triple existed in
    ``source_description``.  The relationship word is located by whole-token
    match over the ENTIRE first body line — never by splitting at ``". "``,
    which truncated entity names containing abbreviations ("Acme Corp. …")
    and silently dropped valid edges.  A line with no vocabulary token, or
    with an empty side of the triple, is not an edge this reader can type.
    """
    head = body.split("\n", 1)[0].strip()
    words = [word for word in head.split(" ") if word]
    vocabulary = {member.value for member in RelationshipType}
    for index, word in enumerate(words):
        token = word.lower().strip(".")
        if token not in vocabulary:
            continue
        from_entity = " ".join(words[:index]).strip().strip(",").removesuffix(".")
        tail = " ".join(words[index + 1 :]).strip()
        to_entity = tail.split(". ", 1)[0].strip().removesuffix(".") if tail else ""
        if from_entity and to_entity:
            return from_entity, token, to_entity
        return None
    return None


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
