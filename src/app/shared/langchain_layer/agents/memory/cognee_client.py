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

import os
from typing import TYPE_CHECKING, override

import cognee
from cognee.exceptions import CogneeApiError
from langgraph.store.base import BaseStore
from pydantic import BaseModel, ConfigDict

from app.connections.postgres import get_database_fields
from app.features.documents.model import CHUNK_EMBEDDING_DIM
from app.utils import logger

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from app.config import Settings


# ---------------------------------------------------------------------------
# Factory: Setup Cognee with Neo4j + Postgres
# ---------------------------------------------------------------------------

_ACCESS_CONTROL_ENV_KEY = "ENABLE_BACKEND_ACCESS_CONTROL"

# Placeholder defaults from the settings surface. A connection field still
# holding one of these is a configuration that was never made, and using it
# would point agent memory at nothing.
_PLACEHOLDER_VALUES = frozenset({"localhost", "db", "user"})


class CogneeSetupError(RuntimeError):
    """Named configuration failure — the memory subsystem cannot start as configured."""


class CogneeDimensionMismatchError(CogneeSetupError):
    """Memory embeddings would not match document embeddings. Boot-stopping."""


class CogneeSetupConfig(BaseModel):
    """Typed result of :func:`setup_cognee`.

    Deliberately carries no credential-shaped field: this object reaches
    ``app.state.cognee_config``, where any string that looked like a URL or a
    password would be one log line away from leaking.
    """

    model_config = ConfigDict(frozen=True)

    service: str = "cognee"
    llm_model: str
    embedding_model: str
    embedding_dimension: int
    neo4j_uri: str
    postgres_host: str
    postgres_database: str
    vector_provider: str
    schema_name: str
    access_control_enabled: bool


async def setup_cognee(settings: Settings) -> CogneeSetupConfig:
    """Configure Cognee to use the same Neo4j + Postgres as the app.

    Cognee creates its own internal connection pools — these are separate
    from app.state.neo4j_driver and app.state.db_engine. The databases
    are the same; the pools are distinct.

    Args:
        settings: Application settings with Neo4j + Postgres credentials.

    Returns:
        A typed, credential-free configuration summary.

    Raises:
        CogneeDimensionMismatchError: If the memory embedding dimension does not
            equal the document embedding dimension. Decision 15 hard-fail class:
            this must stop the boot, not degrade.
        CogneeSetupError: On divergent or placeholder connection settings.
    """
    logger.bind(service="cognee").info("Configuring Cognee")

    # Task 4.2 — BEFORE the first set_*_config call: left unset, cognee's default
    # branch reaches multi_user_support_possible() and raises EnvironmentError on
    # this repository's handler/provider pair (Decision 6).
    os.environ[_ACCESS_CONTROL_ENV_KEY] = (
        "true" if settings.COGNEE_ACCESS_CONTROL_ENABLED else "false"
    )

    # Task 4.5 — the relational identity comes from the application's own accessor,
    # so agent memory and the application pool cannot disagree silently.
    database_fields = get_database_fields()
    if database_fields.host in _PLACEHOLDER_VALUES or database_fields.database in (
        _PLACEHOLDER_VALUES
    ):
        msg = (
            "Postgres connection fields still hold placeholder defaults; refusing to "
            f"point agent memory at host={database_fields.host!r} db={database_fields.database!r}"
        )
        raise CogneeSetupError(msg)
    if (
        database_fields.host != settings.POSTGRES_HOST
        or database_fields.port != settings.POSTGRES_PORT
        or database_fields.database != settings.POSTGRES_DB_NAME
    ):
        msg = (
            "Discrete Postgres settings disagree with POSTGRES_URL — agent memory would "
            "connect to a different database than the application"
        )
        raise CogneeSetupError(msg)

    # Task 4.3 — memory embeddings must be comparable with document embeddings;
    # pin both model and dimension, and refuse to boot on divergence.
    expected_dimension = CHUNK_EMBEDDING_DIM
    if expected_dimension != settings.EMBEDDING_DIMENSION:
        msg = (
            f"EMBEDDING_DIMENSION={settings.EMBEDDING_DIMENSION} but the document "
            f"embedder stores width {expected_dimension}; agent memory would embed "
            "into an incomparable space"
        )
        raise CogneeDimensionMismatchError(msg)

    try:
        cognee.config.set_llm_config(
            config_dict={
                "llm_provider": "google_genai",
                "llm_model": settings.GEMINI_FLASH_MODEL,
                "llm_api_key": settings.GEMINI_API_KEY.get_secret_value(),
            }
        )
        cognee.config.set_embedding_config(
            config_dict={
                "embedding_provider": "google_genai",
                "embedding_model": settings.GEMINI_EMBEDDING_MODEL,
                "embedding_dimensions": settings.EMBEDDING_DIMENSION,
                "embedding_api_key": settings.GEMINI_API_KEY.get_secret_value(),
            }
        )
        cognee.config.set_graph_db_config(
            {
                "graph_database_provider": "neo4j",
                "graph_database_url": settings.NEO4J_URI,
                "graph_database_username": settings.NEO4J_USERNAME,
                "graph_database_password": settings.NEO4J_PASSWORD.get_secret_value(),
            }
        )
        cognee.config.set_relational_db_config(
            {
                "db_provider": "postgres",
                "db_host": database_fields.host,
                "db_port": str(database_fields.port),
                "db_username": database_fields.username,
                "db_password": database_fields.password.get_secret_value(),
                "db_name": database_fields.database,
                "db_path": "",
            }
        )
        # Task 4.4 — explicit vector store; the library default writes embeddings
        # to local files, which is defect two of item 152.
        cognee.config.set_vector_db_config(
            {"vector_db_provider": settings.COGNEE_VECTOR_PROVIDER}
        )
    except Exception:
        logger.bind(service="cognee").exception("Failed to configure Cognee")
        raise
    else:
        logger.bind(service="cognee").info("Cognee configured successfully")
        return CogneeSetupConfig(
            llm_model=settings.GEMINI_FLASH_MODEL,
            embedding_model=settings.GEMINI_EMBEDDING_MODEL,
            embedding_dimension=settings.EMBEDDING_DIMENSION,
            neo4j_uri=settings.NEO4J_URI,
            postgres_host=database_fields.host,
            postgres_database=database_fields.database,
            vector_provider=settings.COGNEE_VECTOR_PROVIDER,
            schema_name=settings.COGNEE_DB_SCHEMA,
            access_control_enabled=settings.COGNEE_ACCESS_CONTROL_ENABLED,
        )


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
    After remember() + improve(), the report becomes queryable via
    recall() for future context retrieval.

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
        await cognee.remember(report_json, dataset_name=dataset_name)
        await cognee.improve(dataset=dataset_name)
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
        ).info("Cognee improve completed successfully")


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
        await cognee.remember(relationships_text, dataset_name=dataset_name)
        await cognee.improve(dataset=dataset_name)
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
# READ: recall episodic memory
# ---------------------------------------------------------------------------


async def search_episodic_memory(
    query: str,
    user_id: str,
) -> list[dict[str, Any]]:
    """Retrieve relevant past decisions from Cognee's episodic memory.

    Uses recall() with auto-routing — returns structured knowledge,
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

        results = await cognee.recall(
            query_text=query,
            datasets=[dataset_name],
        )
    except CogneeApiError as exc:
        exc.add_note(f"query={query[:80]}, user_id={user_id}")
        logger.bind(
            service="cognee",
            query=query,
            user_id=user_id,
        ).exception("Cognee recall failed")
        return []
    else:
        result_list = [dict(r) for r in (results or [])]
        logger.bind(
            service="cognee",
            result_count=len(result_list),
            user_id=user_id,
        ).info("Cognee recall completed successfully")
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

    @override
    async def put(  # ty: ignore[invalid-method-override]
        self,
        namespace: Sequence[str | None],
        key: str,
        value: Any,  # ARG002
    ) -> None:
        """Store a value in the graph with embeddings."""

    @override
    async def get(  # ty: ignore[invalid-method-override]
        self,
        namespace: Sequence[str | None],
        key: str,
    ) -> Any | None:
        """Retrieve a value by namespace + key."""
        return None

    @override
    async def search(  # ty: ignore[invalid-method-override]
        self,
        _namespace: Sequence[str | None],
        *,
        filter_query: dict[str, Any] | None = None,
        query: str | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        """Semantic search within a namespace with optional filtering."""
        return []

    @override
    async def delete(  # ty: ignore[invalid-method-override]
        self,
        namespace: Sequence[str | None],
        *,
        key: str | None = None,
    ) -> None:
        """Delete a single key or entire namespace."""

    async def list_keys(  # noqa: PLR6301 — placeholder stub, interface-shape only
        self,
        namespace: Sequence[str | None],  # noqa: ARG002
    ) -> list[str]:
        """List all keys in a namespace."""
        return []

    @staticmethod
    def _make_key(namespace: Sequence[str | None], key: str) -> str:
        """Construct a full key from namespace hierarchy + key."""
        return "/".join(filter(None, [*namespace, key]))

    @staticmethod
    def _matches_filter(data: Any, filter_dict: dict[str, Any]) -> bool:
        """Check if data matches the filter criteria."""
        if not isinstance(data, dict):
            return False
        return all(data.get(k) == v for k, v in filter_dict.items())
