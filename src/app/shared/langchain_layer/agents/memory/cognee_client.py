"""
Cognee: long-term episodic + procedural memory for Agent Saul.

Cognee role:
  - Episodic: store final approved reports → queryable as "what did I review before?"
  - Procedural: store workflow patterns → "how was this clause type resolved before?"

Cognee reuses the SAME Neo4j and Postgres instances as the rest of the app.
Configuration uses the same env vars / settings — Cognee creates its own
internal connection pools but points at the same database endpoints.

Cognee does NOT replace Graphiti:
  Graphiti → structural legal knowledge graph (clause → relationship → entity)
  Cognee   → episodic + procedural memory (final reports, workflow history)

Initialization (in lifespan.py):
    from src.app.shared.langchain_layer.agents.memory import setup_cognee

    cognee_config = await setup_cognee(settings)
    app.state.cognee_config = cognee_config

    # On shutdown (no explicit close needed — Cognee manages its own pools)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cognee
from cognee import SearchType
from langgraph.store.base import BaseStore

from app.utils import logger

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from app.config import Settings


# ---------------------------------------------------------------------------
# Factory: Setup Cognee with Neo4j + Postgres
# ---------------------------------------------------------------------------


async def setup_cognee(settings: Settings) -> dict:
    """Configure Cognee to use the same Neo4j + Postgres as the app.

    Cognee creates its own internal connection pools — these are separate
    from app.state.neo4j_driver and app.state.db_engine. The databases
    are the same; the pools are distinct.

    Args:
        settings: Application settings with Neo4j + Postgres credentials.

    Returns:
        Configuration dict for this session (for reference/logging).

    Raises:
        Exception: If configuration fails.
    """
    logger.bind(service="cognee").info("Configuring Cognee")

    try:
        cognee.config.set_llm_config(
            config_dict={
                "llm_provider": "google_genai",
                "llm_model": settings.GEMINI_FLASH_MODEL,
                "llm_api_key": settings.GOOGLE_API_KEY,
            }
        )
        cognee.config.set_graph_db_config(
            {
                "graph_database_provider": "neo4j",
                "graph_database_url": settings.NEO4J_URI,
                "graph_database_username": settings.NEO4J_USERNAME,
                "graph_database_password": settings.NEO4J_PASSWORD,
            }
        )
        cognee.config.set_relational_db_config(
            {
                "db_provider": "postgres",
                "db_host": settings.POSTGRES_HOST,
                "db_port": str(settings.POSTGRES_PORT),
                "db_username": settings.POSTGRES_USERNAME,
                "db_password": settings.POSTGRES_PASSWORD,
                "db_name": settings.POSTGRES_DB_NAME,
                "db_path": "",
            }
        )
    except Exception:
        logger.bind(service="cognee").exception("Failed to configure Cognee")
        raise
    else:
        config = {
            "service": "cognee",
            "llm_model": settings.GEMINI_FLASH_MODEL,
            "neo4j_uri": settings.NEO4J_URI,
            "postgres_url": settings.POSTGRES_URL,
        }
        logger.bind(service="cognee").info("Cognee configured successfully")
        return config


# ---------------------------------------------------------------------------
# WRITE: store final report (episodic memory)
# ---------------------------------------------------------------------------


async def store_final_report(
    report_json: str,
    user_id: str,
    doc_id: str,
    thread_id: str,
) -> None:
    """Store a human-approved final report in Cognee's episodic memory.

    Dataset name: {user_id}.legal_reports
    After add() + cognify(), the report becomes queryable via
    search(SearchType.INSIGHTS, ...) for future context retrieval.

    Args:
        report_json: JSON-serialized final report.
        user_id: User ID for dataset namespacing.
        doc_id: Document ID for tracing.
        thread_id: Thread ID for tracing.
    """
    dataset_name = f"{user_id}.legal_reports"
    logger.bind(
        service="cognee",
        dataset_name=dataset_name,
        doc_id=doc_id,
        user_id=user_id,
        thread_id=thread_id,
    ).info("Storing final report in Cognee")

    try:
        await cognee.add(report_json, dataset_name=dataset_name)
        await cognee.cognify(datasets=[dataset_name])
    except Exception:
        logger.bind(
            service="cognee",
            dataset_name=dataset_name,
            doc_id=doc_id,
            thread_id=thread_id,
        ).exception("Failed to store final report in Cognee")
        raise
    else:
        logger.bind(
            service="cognee",
            dataset_name=dataset_name,
            doc_id=doc_id,
            thread_id=thread_id,
        ).info("Cognee cognify completed successfully")


# ---------------------------------------------------------------------------
# WRITE: store relationship graph (procedural memory)
# ---------------------------------------------------------------------------


async def store_relationships(
    relationships_text: str,
    user_id: str,
    doc_id: str,
) -> None:
    """Store the legal relationship graph summary in Cognee.

    Stored in {user_id}.legal_relationships — allows querying
    patterns like 'contracts where Party X has unlimited liability'.

    Args:
        relationships_text: Relationship graph summary text.
        user_id: User ID for dataset namespacing.
        doc_id: Document ID for tracing.
    """
    dataset_name = f"{user_id}.legal_relationships"
    logger.bind(
        service="cognee",
        dataset_name=dataset_name,
        doc_id=doc_id,
        user_id=user_id,
    ).info("Storing relationships in Cognee")

    try:
        await cognee.add(relationships_text, dataset_name=dataset_name)
        await cognee.cognify(datasets=[dataset_name])
    except Exception:
        logger.bind(
            service="cognee",
            dataset_name=dataset_name,
            doc_id=doc_id,
        ).exception("Failed to store relationships in Cognee")
        raise
    else:
        logger.bind(
            service="cognee",
            dataset_name=dataset_name,
            doc_id=doc_id,
        ).info("Cognee relationships stored successfully")


# ---------------------------------------------------------------------------
# READ: search episodic memory
# ---------------------------------------------------------------------------


async def search_episodic_memory(
    query: str,
    user_id: str,
) -> list[dict[str, Any]]:
    """Retrieve relevant past decisions from Cognee's episodic memory.

    Uses INSIGHTS search type — returns structured knowledge,
    not raw chunks. Returns empty list on any failure so callers
    can degrade gracefully.

    Args:
        query: Search query string.
        user_id: User ID to scope search to their memories.

    Returns:
        List of search results as dicts, empty list on failure.
    """
    try:
        dataset_name = f"{user_id}.legal_reports"
        logger.bind(
            service="cognee",
            query=query,
            user_id=user_id,
            dataset_name=dataset_name,
        ).info("Searching Cognee episodic memory")

        results = await cognee.search(  # type: ignore[attr-defined]
            SearchType.INSIGHTS,  # type: ignore[attr-defined]
            query=query,  # type: ignore[call-arg]
            datasets=[dataset_name],
        )
    except Exception:  # noqa: BLE001
        logger.bind(
            service="cognee",
            query=query,
            user_id=user_id,
        ).exception("Cognee search failed")
        return []
    else:
        result_list = [dict(r) for r in (results or [])]
        logger.bind(
            service="cognee",
            result_count=len(result_list),
            user_id=user_id,
        ).info("Cognee search completed successfully")
        return result_list


# ---------------------------------------------------------------------------
# BaseStore Implementation for LangGraph integration
# ---------------------------------------------------------------------------


class CogneeStore(BaseStore):
    """Cognee-backed store for LangGraph persistent memory.

    Implements BaseStore protocol to work with LangGraph's Store
    for long-term thread memory across graph invocations.

    Note: Placeholder implementation — adapt cognee API calls to actual library.
    """

    def __init__(self, cognee_client: Any) -> None:
        self.client = cognee_client

    async def put(  # type: ignore[override]
        self,
        namespace: Sequence[str | None],
        key: str,
        value: Any,  # ARG002
    ) -> None:
        """Store a value in the graph with embeddings."""

    async def get(  # type: ignore[override]
        self,
        namespace: Sequence[str | None],
        key: str,  # noqa: ARG002
    ) -> Any | None:
        """Retrieve a value by namespace + key."""
        return None

    async def search(  # type: ignore[override]
        self,
        _namespace: Sequence[str | None],
        *,
        filter: dict | None = None,  # noqa: A002, ARG002
        query: str | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> list[Any]:
        """Semantic search within a namespace with optional filtering."""
        return []

    async def delete(  # type: ignore[override]
        self,
        namespace: Sequence[str | None],
        *,
        key: str | None = None,
    ) -> None:
        """Delete a single key or entire namespace."""

    async def list_keys(  # type: ignore[override]
        self,
        namespace: Sequence[str | None],  # noqa: ARG002
    ) -> list[str]:
        """List all keys in a namespace."""
        return []

    def _make_key(self, namespace: Sequence[str | None], key: str) -> str:
        """Construct a full key from namespace hierarchy + key."""
        return "/".join(filter(None, [*namespace, key]))

    def _matches_filter(self, data: Any, filter_dict: dict) -> bool:  # type: ignore[name-defined]
        """Check if data matches the filter criteria."""
        if not isinstance(data, dict):
            return False
        return all(data.get(k) == v for k, v in filter_dict.items())
