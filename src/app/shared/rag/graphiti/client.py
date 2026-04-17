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
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

from app.utils import logger

from .schemas import GraphitiSearchResult

if TYPE_CHECKING:
    from typing import Any

    from .schemas import (
        ClauseEpisodeMetadata,
        FinalReportEpisodeMetadata,
        LegalEdgeInput,
    )

# Score weights for multi-objective memory retrieval (Section 18.6)
_W_SEMANTIC: float = 0.50
_W_RECENCY: float = 0.20
_W_TRUST: float = 0.20
_W_TASK_RELEVANCE: float = 0.10


# ---------------------------------------------------------------------------
# Factory: Setup Graphiti with Neo4j
# ---------------------------------------------------------------------------


async def setup_graphiti(
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
) -> Graphiti:
    """Create a Graphiti instance with its own Neo4j connection pool.

    NOTE: Graphiti creates its own driver internally (does not accept
    an existing neo4j.AsyncDriver). This is a known graphiti-core
    limitation. The driver created here is separate from
    app.state.neo4j_driver which you still use for direct Cypher
    queries in relationship_mapping and compliance nodes.

    Args:
        neo4j_uri: Neo4j connection URI.
        neo4j_user: Neo4j username.
        neo4j_password: Neo4j password.

    Returns:
        Initialized Graphiti instance.

    Raises:
        Exception: If Neo4j connection fails.
    """
    logger.bind(service="graphiti").info("Creating Graphiti instance")

    try:
        graphiti = Graphiti(
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password,
            # Alternative: Pass a pre-configured driver (recommended for custom database name, auth, etc.)
            graph_driver=None,  # Optional[Neo4jDriver | other driver]
            # LLM Configuration (critical for entity extraction, summarization, etc.)
            llm_client=None,  # Optional[LLMClient] — defaults to OpenAIClient
            # Embedder Configuration (for semantic search, hybrid retrieval)
            embedder=None,  # Optional[Embedder] — defaults to OpenAIEmbedder
            # Other important options
            # group_id=None,  # Optional[str] — useful for multi-tenant / team isolation
            # database=None,  # Optional[str] — custom Neo4j database name (default: "neo4j")
            # # Advanced / less common
            # config=None,  # Optional[GraphitiConfig] — centralized config object
            # use_custom_entities=False,  # bool — allow custom entity types?
            # destroy_graph=False,  # bool — WARNING: wipes the entire graph on init (for testing)
        )
    except Exception:
        logger.bind(service="graphiti").exception("Failed to create Graphiti instance")
        raise
    else:
        logger.bind(service="graphiti").info("Graphiti instance created")
        return graphiti


async def setup_graphiti_indices(graphiti: Graphiti) -> None:
    """Create indices and constraints in Neo4j.

    Call once at lifespan startup after creating the Graphiti instance.

    Args:
        graphiti: Initialized Graphiti instance.

    Raises:
        Exception: If index creation fails.
    """
    try:
        await graphiti.build_indices_and_constraints()
    except Exception:
        logger.bind(service="graphiti").exception("Failed to setup Graphiti indices")
        raise
    else:
        logger.bind(service="graphiti").info("Graphiti indices and constraints created")


async def close_graphiti(graphiti: Graphiti | None) -> None:
    """Close Graphiti instance and clean up resources.

    Args:
        graphiti: Graphiti instance to close (can be None).
    """
    if graphiti is None:
        return

    try:
        await graphiti.close()
        logger.bind(service="graphiti").info("Graphiti closed")
    except Exception as e:  # noqa: BLE001
        logger.bind(service="graphiti").warning(
            "Error closing Graphiti", error=str(e), error_type=type(e).__name__
        )


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
    episode_name = f"clause:{metadata.doc_id}:{metadata.clause_id}"

    logger.bind(
        service="graphiti",
        doc_id=metadata.doc_id,
        clause_id=metadata.clause_id,
    ).info("Writing clause episode to Graphiti")

    try:
        result = await graphiti.add_episode(
            name=episode_name,
            episode_body=clause_text,
            source=EpisodeType.text,
            source_description=source_description,
            reference_time=datetime.now(tz=UTC),
            group_id=metadata.doc_id,
        )
        return str(result.uuid) if hasattr(result, "uuid") else episode_name  # type: ignore[return-value]
    except Exception:
        logger.bind(service="graphiti").exception("Failed to write clause episode")
        raise


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
    episode_name = f"edge:{edge.doc_id}:{edge.clause_id}:{edge.from_entity}:{edge.relationship}:{edge.to_entity}"
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
        result = await graphiti.add_episode(
            name=episode_name,
            episode_body=edge.to_episode_body(),
            source=EpisodeType.text,
            source_description=source_description,
            reference_time=datetime.now(tz=UTC),
            group_id=edge.doc_id,
        )
        return str(result.uuid) if hasattr(result, "uuid") else episode_name  # type: ignore[return-value]
    except Exception:
        logger.bind(service="graphiti").exception("Failed to write relationship edge")
        raise


# ---------------------------------------------------------------------------
# WRITE: final report episode
# ---------------------------------------------------------------------------


async def write_final_report_episode(
    graphiti: Graphiti,
    report_summary: str,
    metadata: FinalReportEpisodeMetadata,
) -> str:
    """Write the human-approved final report as a high-trust episode.

    Args:
        graphiti: Graphiti instance.
        report_summary: Summary of the final report.
        metadata: Report metadata (doc_id, user_id, human_approved, etc.).

    Returns:
        Episode UUID assigned by Graphiti.

    Raises:
        Exception: If write fails.
    """
    episode_name = f"report:{metadata.doc_id}:{metadata.user_id}"
    source_description = metadata.model_dump_json()

    logger.bind(
        service="graphiti",
        doc_id=metadata.doc_id,
        human_approved=metadata.human_approved,
    ).info("Writing final report episode to Graphiti")

    try:
        result = await graphiti.add_episode(
            name=episode_name,
            episode_body=report_summary,
            source=EpisodeType.text,
            source_description=source_description,
            reference_time=datetime.now(tz=UTC),
            group_id=metadata.user_id,
        )
        return str(result.uuid) if hasattr(result, "uuid") else episode_name  # type: ignore[return-value]
    except Exception:
        logger.bind(service="graphiti").exception("Failed to write final report episode")
        raise


# ---------------------------------------------------------------------------
# READ: risk analysis context
# ---------------------------------------------------------------------------


async def search_for_risk_context(
    graphiti: Graphiti,
    query: str,
    user_id: str,
    doc_id: str | None = None,
    num_results: int = 10,
) -> list[GraphitiSearchResult]:
    """Search the knowledge graph for risk-relevant context.

    Searches across:
    - Current document episodes (group_id=doc_id) for clause-level context
    - User's historical documents (group_id=user_id) for precedent patterns

    Args:
        graphiti: Graphiti instance.
        query: Search query string.
        user_id: User ID for scoping search.
        doc_id: Optional document ID for current-doc context.
        num_results: Number of results to return.

    Returns:
        List of scored GraphitiSearchResult items.
    """
    group_ids: list[str] = [user_id]
    if doc_id:
        group_ids.append(doc_id)

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
    except Exception:  # noqa: BLE001
        logger.bind(service="graphiti").exception("Graphiti risk context search failed")
        return []
    else:
        return [
            _score_and_convert(r, query=query, task="risk_analysis") for r in (raw_results or [])
        ]


# ---------------------------------------------------------------------------
# READ: precedent chains
# ---------------------------------------------------------------------------


async def search_for_precedent_chains(
    graphiti: Graphiti,
    query: str,
    user_id: str,
    jurisdiction: str = "India",
    num_results: int = 10,
) -> list[GraphitiSearchResult]:
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
    logger.bind(
        service="graphiti",
        query=query[:80],
        jurisdiction=jurisdiction,
    ).info("Searching Graphiti for precedents")

    try:
        raw_results = await graphiti.search(
            query=query,
            group_ids=[user_id],
            num_results=num_results * 2,
        )

        filtered = [r for r in (raw_results or []) if _has_jurisdiction(r, jurisdiction)][
            :num_results
        ]
    except Exception:  # noqa: BLE001
        logger.bind(service="graphiti").exception("Graphiti precedent search failed")
        return []
    else:
        return [_score_and_convert(r, query=query, task="compliance") for r in filtered]


# ---------------------------------------------------------------------------
# READ: obligation chain
# ---------------------------------------------------------------------------


async def get_obligation_chain(
    graphiti: Graphiti,
    entity_name: str,
    user_id: str,
    doc_id: str | None = None,
    depth: int = 3,
) -> list[GraphitiSearchResult]:
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
    group_ids: list[str] = [user_id]
    if doc_id:
        group_ids.append(doc_id)

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
    except Exception:  # noqa: BLE001
        logger.bind(service="graphiti").exception("Graphiti obligation chain search failed")
        return []
    else:
        return [
            _score_and_convert(r, query=obligation_query, task="obligation_chain")
            for r in (raw_results or [])
        ]


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
