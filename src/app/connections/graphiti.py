"""Knowledge-graph connection establishment (real implementation, not a seam).

The lifecycle functions live HERE, in the connections package, because they
are what they say: external-system adapters. The domain RAG behavior that
consumes a live Graphiti instance (the bound service, schemas, subgraph
loading, clause-episode writers) lives in ``app.retrieval.graphiti`` —
multiple feature importers reach it directly.

Split contracts (import-linter):
- ``app.connections`` stays feature-free and worker-free here.
- ``app.retrieval`` stays feature-free except the one documented scheduled
  move (the clause-episode writer's clause-type protocol), which this change
  did not relocate.

Lifespan usage:
    graphiti = await setup_graphiti(
        neo4j_uri=settings.NEO4J_URI,
        neo4j_user=settings.NEO4J_USERNAME,
        neo4j_password=settings.NEO4J_PASSWORD,
        neo4j_database=settings.NEO4J_DATABASE,
    )
    await setup_graphiti_indices(graphiti)
    app.state.graphiti = graphiti
"""

from __future__ import annotations

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.errors import GraphitiError
from graphiti_core.llm_client.gemini_client import GeminiClient, LLMConfig

from app.config import get_settings
from app.observability.logging import logger

DEFAULT_NEO4J_DATABASE = "neo4j"

settings = get_settings()


async def setup_graphiti(
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    neo4j_database: str | None = None,
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
        neo4j_database: Neo4j database name (Aura instances are not "neo4j").

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
            # Pass the process-owned driver: Graphiti's own construction
            # hard-codes database="neo4j", which breaks on Aura instances
            # whose only user database is named after the instance id. The
            # caller's configured database name (NEO4J_DATABASE) rules.
            graph_driver=Neo4jDriver(
                uri=neo4j_uri,
                user=neo4j_user,
                password=neo4j_password,
                database=neo4j_database or DEFAULT_NEO4J_DATABASE,
            ),
            # LLM Configuration (critical for entity extraction, summarization, etc.)
            llm_client=GeminiClient(
                config=LLMConfig(
                    api_key=settings.GEMINI_API_KEY.get_secret_value(),
                    model=settings.GEMINI_FLASH_MODEL,
                )
            ),
            embedder=GeminiEmbedder(
                config=GeminiEmbedderConfig(
                    api_key=settings.GEMINI_API_KEY.get_secret_value(),
                    embedding_model=settings.GEMINI_EMBEDDING_MODEL,
                )
            ),
            cross_encoder=GeminiRerankerClient(
                config=LLMConfig(
                    api_key=settings.GEMINI_API_KEY.get_secret_value(),
                    model=settings.GEMINI_FLASH_MODEL,
                )
            ),
            # Other important options
            # group_id=None,  # Optional[str] — useful for multi-tenant / team isolation
            # database=None,  # Optional[str] — custom Neo4j database name (default: "neo4j")
            # # Advanced / less common
            # config=None,  # Optional[GraphitiConfig] — centralized config object
            # use_custom_entities=False,  # bool — allow custom entity types?
            # destroy_graph=False,  # bool — WARNING: wipes the entire graph on init (for testing)
        )
    except Exception as exc:
        exc.add_note("operation=create_graphiti, service=graphiti")
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
    except Exception as exc:
        exc.add_note("operation=setup_graphiti_indices, service=graphiti")
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
    except GraphitiError as e:
        e.add_note("operation=close_graphiti")
        logger.bind(service="graphiti").warning(
            "Error closing Graphiti", error=str(e), error_type=type(e).__name__
        )
