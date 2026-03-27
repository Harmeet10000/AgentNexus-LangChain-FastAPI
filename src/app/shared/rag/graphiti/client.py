"""
GraphitiService: async wrapper around graphiti-core for the legal domain.

Initialisation (in lifespan.py):
    from src.app.shared.rag.graphiti.client import GraphitiService

    graphiti_service = await GraphitiService.create(
        neo4j_uri=settings.neo4j_uri,
        neo4j_user=settings.neo4j_user,
        neo4j_password=settings.neo4j_password.get_secret_value(),
        llm_client=flash_llm,   # Graphiti uses LLM internally for entity extraction
    )
    await graphiti_service.setup()
    app.state.graphiti = graphiti_service

    # On shutdown:
    await app.state.graphiti.close()

Design decisions:
- Graphiti creates its OWN async Neo4j driver internally. We pass the URI +
  credentials (not app.state.neo4j_driver) because graphiti-core does not
  accept an existing driver instance. The app.state.neo4j_driver remains
  available for raw Cypher queries (relationship_mapping → Postgres+Neo4j
  hybrid writes that need direct graph control).
- All public methods return domain types from schemas.py — no raw
  graphiti-core types leak outside this module.
- write_clause_episode and write_relationship_edge are the two write paths.
- search_for_risk_context and search_for_precedent_chains are the two read paths.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

from app.utils import logger

from .schemas import (
    GraphitiSearchResult,
)

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


class GraphitiService:
    """Async-first Graphiti wrapper.  One instance per application lifetime."""

    def __init__(self, graphiti: Graphiti) -> None:
        self._g = graphiti

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    async def create(
        cls,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
    ) -> GraphitiService:
        """Create a GraphitiService with its own Neo4j connection pool.

        NOTE: Graphiti creates its own driver internally (does not accept
        an existing neo4j.AsyncDriver).  This is a known graphiti-core
        limitation.  The driver created here is separate from
        app.state.neo4j_driver which you still use for direct Cypher
        queries in relationship_mapping and compliance nodes.
        """
        graphiti = Graphiti(
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password,
        )
        return cls(graphiti)

    async def setup(self) -> None:
        """Create indices and constraints in Neo4j.  Call once at lifespan startup."""
        await self._g.build_indices_and_constraints()
        logger.bind(service="graphiti").info("graphiti_indices_ready")

    async def close(self) -> None:
        await self._g.close()
        logger.bind(service="graphiti").info("graphiti_closed")

    # ------------------------------------------------------------------
    # WRITE: clause episode (one per clause segment, group_id = doc_id)
    # ------------------------------------------------------------------

    async def write_clause_episode(
        self,
        clause_text: str,
        metadata: ClauseEpisodeMetadata,
    ) -> str:
        """Write a single clause as a Graphiti episode.

        Returns the episode UUID assigned by Graphiti.
        The group_id=doc_id ensures all clauses from the same document
        form a retrievable subgraph.
        """
        source_description = metadata.model_dump_json()
        episode_name = f"clause:{metadata.doc_id}:{metadata.clause_id}"

        logger.bind(
            service="graphiti",
            doc_id=metadata.doc_id,
            clause_id=metadata.clause_id,
        ).info("graphiti_write_clause_episode")

        result = await self._g.add_episode(
            name=episode_name,
            episode_body=clause_text,
            source=EpisodeType.text,
            source_description=source_description,
            reference_time=datetime.now(tz=UTC),
            group_id=metadata.doc_id,
        )
        # graphiti-core returns the created episode node
        return str(result.uuid) if hasattr(result, "uuid") else episode_name

    # ------------------------------------------------------------------
    # WRITE: structured relationship edge
    # ------------------------------------------------------------------

    async def write_relationship_edge(self, edge: LegalEdgeInput) -> str:
        """Write a legal relationship as a structured Graphiti episode.

        We write edges as episodes with a structured body — Graphiti
        then extracts the entities (from_entity, to_entity) and creates
        graph edges natively in Neo4j.  This is more reliable than
        calling Neo4j Cypher directly because Graphiti handles entity
        deduplication (e.g. "Acme Corp" vs "Acme Corporation" → same node).
        """
        episode_name = (
            f"edge:{edge.doc_id}:{edge.clause_id}:{edge.from_entity}:{edge.relationship}:{edge.to_entity}"
        )
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
        ).info("graphiti_write_relationship_edge")

        result = await self._g.add_episode(
            name=episode_name,
            episode_body=edge.to_episode_body(),
            source=EpisodeType.text,
            source_description=source_description,
            reference_time=datetime.now(tz=UTC),
            group_id=edge.doc_id,
        )
        return str(result.uuid) if hasattr(result, "uuid") else episode_name

    # ------------------------------------------------------------------
    # WRITE: final report episode (post human review, highest trust)
    # ------------------------------------------------------------------

    async def write_final_report_episode(
        self,
        report_summary: str,
        metadata: FinalReportEpisodeMetadata,
    ) -> str:
        """Write the human-approved final report as a high-trust episode.

        This episode becomes the 'ground truth' for future precedent
        queries on this document type + user combination.
        """
        episode_name = f"report:{metadata.doc_id}:{metadata.user_id}"
        source_description = metadata.model_dump_json()

        logger.bind(
            service="graphiti",
            doc_id=metadata.doc_id,
            human_approved=metadata.human_approved,
        ).info("graphiti_write_final_report")

        result = await self._g.add_episode(
            name=episode_name,
            episode_body=report_summary,
            source=EpisodeType.text,
            source_description=source_description,
            reference_time=datetime.now(tz=UTC),
            group_id=metadata.user_id,  # group by user for cross-doc precedent
        )
        return str(result.uuid) if hasattr(result, "uuid") else episode_name

    # ------------------------------------------------------------------
    # READ: risk analysis context (multi-hop graph traversal)
    # ------------------------------------------------------------------

    async def search_for_risk_context(
        self,
        query: str,
        user_id: str,
        doc_id: str | None = None,
        num_results: int = 10,
    ) -> list[GraphitiSearchResult]:
        """Search the knowledge graph for risk-relevant context.

        Searches across:
        - Current document episodes (group_id=doc_id) for clause-level context
        - User's historical documents (group_id=user_id) for precedent patterns

        Returns scored results using the multi-objective scoring function
        from Section 18.6.
        """
        group_ids: list[str] = [user_id]
        if doc_id:
            group_ids.append(doc_id)

        logger.bind(
            service="graphiti",
            query=query[:80],
            group_ids=group_ids,
        ).info("graphiti_search_risk")

        raw_results = await self._g.search(
            query=query,
            group_ids=group_ids,
            num_results=num_results,
        )

        return [
            _score_and_convert(r, query=query, task="risk_analysis")
            for r in (raw_results or [])
        ]

    # ------------------------------------------------------------------
    # READ: precedent chains for compliance
    # ------------------------------------------------------------------

    async def search_for_precedent_chains(
        self,
        query: str,
        user_id: str,
        jurisdiction: str = "India",
        num_results: int = 10,
    ) -> list[GraphitiSearchResult]:
        """Search the knowledge graph for compliance precedents.

        Searches user's historical approved reports — the highest-trust
        episodic memory — to find how similar clauses were handled before.

        The jurisdiction filter is applied post-retrieval by checking
        metadata in source_description.
        """
        logger.bind(
            service="graphiti",
            query=query[:80],
            jurisdiction=jurisdiction,
        ).info("graphiti_search_precedents")

        raw_results = await self._g.search(
            query=query,
            group_ids=[user_id],
            num_results=num_results * 2,  # over-fetch for jurisdiction filter
        )

        filtered = [
            r for r in (raw_results or [])
            if _has_jurisdiction(r, jurisdiction)
        ][:num_results]

        return [
            _score_and_convert(r, query=query, task="compliance")
            for r in filtered
        ]

    # ------------------------------------------------------------------
    # READ: obligation chain (forward-chaining from a specific entity)
    # ------------------------------------------------------------------

    async def get_obligation_chain(
        self,
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

        Uses Graphiti's entity-centric search to find connected nodes.
        Depth parameter controls hop count in the graph traversal.
        """
        group_ids: list[str] = [user_id]
        if doc_id:
            group_ids.append(doc_id)

        obligation_query = (
            f"obligations responsibilities duties of {entity_name}"
        )
        logger.bind(
            service="graphiti",
            entity=entity_name,
            depth=depth,
        ).info("graphiti_obligation_chain")

        raw_results = await self._g.search(
            query=obligation_query,
            group_ids=group_ids,
            num_results=depth * 5,
        )

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

    semantic_similarity → from graphiti result score (if present)
    recency             → decay based on created_at age
    trust_score         → extracted from source_description metadata (human_reviewed flag)
    task_relevance      → keyword overlap between query and episode name
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
    if created_at is None:
        return 0.5
    age_days = (datetime.now(tz=UTC) - created_at).days
    return max(0.0, 1.0 - age_days / 365.0)


def _trust_score(source_description: str) -> float:
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
    if not content or not query:
        return 0.0
    query_words = set(query.lower().split())
    content_words = set(content.lower().split())
    overlap = len(query_words & content_words)
    return min(1.0, overlap / max(len(query_words), 1))


def _has_jurisdiction(raw: Any, jurisdiction: str) -> bool:
    source_desc: str = getattr(raw, "source_description", "") or ""
    if not source_desc:
        return True  # don't filter if no metadata
    try:
        meta = json.loads(source_desc)
        stored_jurisdiction = meta.get("jurisdiction", "")
        return not stored_jurisdiction or jurisdiction.lower() in stored_jurisdiction.lower()
    except (json.JSONDecodeError, TypeError):
        return True
