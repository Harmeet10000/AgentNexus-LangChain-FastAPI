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
from typing import TYPE_CHECKING

import cognee

from app.connections.postgres import get_database_fields as _database_fields
from app.features.documents.model import CHUNK_EMBEDDING_DIM
from app.utils import logger

from .setup_types import CogneeSetupConfig

if TYPE_CHECKING:
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

    # Cognee's schema migrations run in a SUBPROCESS (`python -m alembic`),
    # which inherits only real environment variables — the in-memory
    # set_relational_db_config call below never reaches them. Export the
    # system-database identity so `cognee.run_startup_migrations()` targets
    # the same Postgres as everything else instead of a local sqlite file.
    database_fields = _database_fields()
    os.environ.setdefault("DB_PROVIDER", "postgres")
    os.environ["DB_HOST"] = database_fields.host
    os.environ["DB_PORT"] = str(database_fields.port)
    os.environ["DB_USERNAME"] = database_fields.username
    os.environ["DB_PASSWORD"] = database_fields.password.get_secret_value()
    os.environ["DB_NAME"] = database_fields.database

    # Task 4.5 — the relational identity comes from the application's own accessor,
    # so agent memory and the application pool cannot disagree silently.
    database_fields = _database_fields()
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
    # Imported lazily: a module-level import edges this transport-adjacent module
    # into app.features' package init and creates an import cycle.
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
                "llm_provider": "gemini",
                "llm_model": settings.GEMINI_FLASH_MODEL,
                "llm_api_key": settings.GEMINI_API_KEY.get_secret_value(),
            }
        )
        cognee.config.set_embedding_config(
            config_dict={
                "embedding_provider": "gemini",
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
        cognee.config.set_vector_db_config({"vector_db_provider": settings.COGNEE_VECTOR_PROVIDER})
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
