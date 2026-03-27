"""
CogneeService: wrapper around Cognee for long-term episodic + procedural memory.

Cognee role in Agent Saul:
  - Episodic: store final approved reports → queryable as "what did I review before?"
  - Procedural: store workflow patterns → "how was this clause type resolved before?"

Cognee reuses the SAME Neo4j and Postgres instances as the rest of the app.
Configuration uses the same env vars / settings — Cognee creates its own
internal connection pools but points at the same database endpoints.

Cognee does NOT replace Graphiti.  The split is:
  Graphiti → structural legal knowledge graph (clause → relationship → entity)
  Cognee   → episodic + procedural memory (final reports, workflow history)

Initialisation (in lifespan.py):
    from src.app.shared.services.cognee_service import CogneeService

    cognee_service = await CogneeService.create(
        neo4j_uri=settings.neo4j_uri,
        neo4j_user=settings.neo4j_user,
        neo4j_password=settings.neo4j_password.get_secret_value(),
        db_url=settings.database_url_async,
        llm_model=settings.gemini_flash_model,
        llm_api_key=settings.google_api_key.get_secret_value(),
    )
    app.state.cognee = cognee_service

    # On shutdown (no explicit close needed — Cognee manages its own pools)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cognee
from cognee import SearchType

from app.utils import logger

if TYPE_CHECKING:
    from typing import Any


class CogneeService:
    """Async wrapper around Cognee for Agent Saul's long-term memory.

    All public methods are async and safe to call from LangGraph nodes.
    """

    def __init__(self) -> None:
        self._ready = False

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    async def create(
        cls,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        db_url: str,
        llm_model: str,
        llm_api_key: str,
    ) -> CogneeService:
        """Configure Cognee to use the same Neo4j + Postgres as the app.

        Cognee accepts connection strings / URIs via its config module.
        It creates its own internal pools — these are separate from
        app.state.neo4j_driver and app.state.db_engine.  The databases
        are the same; the pools are distinct.
        """

        # Point Cognee at existing infrastructure
        await cognee.config.set_llm_config(
            {
                "llm_provider": "google_genai",
                "llm_model": llm_model,
                "llm_api_key": llm_api_key,
            }
        )
        await cognee.config.set_graph_db_config(  # type: ignore[attr-defined]
            {
                "graph_db_provider": "neo4j",
                "graph_db_url": neo4j_uri,
                "graph_db_username": neo4j_user,
                "graph_db_password": neo4j_password,
            }
        )
        await cognee.config.set_relational_db_config(
            {
                "db_provider": "postgres",
                "db_connection_string": db_url,
            }
        )

        service = cls()
        service._ready = True
        logger.bind(service="cognee").info("Cognee service initialized successfully")
        return service

    # ------------------------------------------------------------------
    # WRITE: store final report (episodic memory)
    # ------------------------------------------------------------------

    async def store_final_report(
        self,
        report_json: str,
        user_id: str,
        doc_id: str,
        thread_id: str,
    ) -> None:
        """Store a human-approved final report in Cognee's episodic memory.

        Dataset name: {user_id}.legal_reports
        After add() + cognify(), the report becomes queryable via
        search(SearchType.INSIGHTS, ...) for future context retrieval.
        """
        if not self._ready:
            logger.bind(service="cognee").error("Cognee service not ready")
            return

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
            logger.bind(
                service="cognee",
                dataset_name=dataset_name,
                doc_id=doc_id,
                thread_id=thread_id,
            ).info("Cognee cognify completed successfully")
        except Exception:
            logger.bind(
                service="cognee",
                dataset_name=dataset_name,
                doc_id=doc_id,
                thread_id=thread_id,
            ).exception("Failed to store final report in Cognee")
            raise

    # ------------------------------------------------------------------
    # WRITE: store relationship graph (procedural memory)
    # ------------------------------------------------------------------

    async def store_relationships(
        self,
        relationships_text: str,
        user_id: str,
        doc_id: str,
    ) -> None:
        """Store the legal relationship graph summary in Cognee.

        Stored in {user_id}.legal_relationships — allows querying
        patterns like 'contracts where Party X has unlimited liability'.
        """
        if not self._ready:
            logger.bind(service="cognee").error("Cognee service not ready")
            return

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
            logger.bind(
                service="cognee",
                dataset_name=dataset_name,
                doc_id=doc_id,
            ).info("Cognee relationships stored successfully")
        except Exception:
            logger.bind(
                service="cognee",
                dataset_name=dataset_name,
                doc_id=doc_id,
            ).exception("Failed to store relationships in Cognee")
            raise

    # ------------------------------------------------------------------
    # READ: search episodic memory
    # ------------------------------------------------------------------

    async def search_episodic_memory(
        self,
        query: str,
        user_id: str,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant past decisions from Cognee's episodic memory.

        Uses INSIGHTS search type — returns structured knowledge,
        not raw chunks.  Returns empty list on any failure so callers
        can degrade gracefully.
        """
        if not self._ready:
            logger.bind(service="cognee").error("Cognee service not ready")
            return []

        try:
            dataset_name = f"{user_id}.legal_reports"
            logger.bind(
                service="cognee",
                query=query,
                user_id=user_id,
                dataset_name=dataset_name,
            ).info("Searching Cognee episodic memory")

            results = await cognee.search(
                SearchType.INSIGHTS,
                query=query,
                datasets=[dataset_name],
            )
            result_list = [dict(r) for r in (results or [])]
            logger.bind(
                service="cognee",
                result_count=len(result_list),
                user_id=user_id,
            ).info("Cognee search completed successfully")
            return result_list
        except Exception:
            logger.bind(
                service="cognee",
                query=query,
                user_id=user_id,
            ).exception("Cognee search failed")
            return []
