from typing import Any

import cognee
from cognee.modules.search.types import SearchType
from pydantic import BaseModel, Field

from app.utils import ValidationException, logger

# async def initialize_cognee():
#     # Configure Cognee to use your custom infrastructure
#     cognee.config.set(
#         {
#             "graph_database_provider": "neo4j",
#             "graph_database_url": settings.NEO4J_URL,
#             "graph_database_username": settings.NEO4J_USER,
#             "graph_database_password": settings.NEO4J_PASSWORD,
#             "relational_db_provider": "postgres",
#             "relational_db_url": settings.POSTGRES_ASYNC_URL,
#         }
#     )

#     # Optional: Verify connection
#     await cognee.setup()

class SearchQuery(BaseModel):
    text: str = Field(min_length=1, description="The natural language query")
    datasets: list[str] | None = None
    limit: int = Field(default=10, ge=1)


class CognitiveSearchService:
    """
    Modular monolith service for Cognee-powered GraphRAG operations.
    Encapsulates Full-Text, Semantic, Temporal, and Meta-Routing searches.
    """

    async def execute_search(
        self, query: SearchQuery, search_mode: SearchType = SearchType.GRAPH_COMPLETION
    ) -> list[Any]:
        """
        Base execution engine for Cognee search dispatching.
        """
        try:
            logger.info("Executing Cognee search", mode=search_mode, query=query.text)

            # Cognee 2026 handles the hybrid/semantic logic at the retriever layer
            results = await cognee.search(
                query_text=query.text,
                query_type=search_mode,
                datasets=query.datasets,
                top_k=query.limit,
            )
            return results
        except Exception as e:
            logger.error("Cognee search failed", error=str(e))
            raise ValidationException(detail="Failed to retrieve context from memory engine.")

    async def feeling_lucky(self, query: SearchQuery) -> Any:
        """
        Meta-routing search: Uses LLM to determine the optimal SearchType
        (Temporal vs Semantic vs Insight) based on query intent.
        """
        return await self.execute_search(query, SearchType.FEELING_LUCKY)

    async def temporal_search(self, query: SearchQuery) -> Any:
        """
        Time-aware retrieval for 'before', 'after', or specific date ranges.
        """
        return await self.execute_search(query, SearchType.TEMPORAL)

    async def hybrid_search(self, query: SearchQuery) -> Any:
        """
        Combines CHUNKS_LEXICAL (Keyword) with GRAPH_COMPLETION (Semantic).
        """
        # In 2026, INSIGHTS is the preferred hybrid relationship-aware mode
        return await self.execute_search(query, SearchType.INSIGHTS)


from langgraph.store.base import BaseStore
from typing import Any, Optional, Sequence
import cognee as cg  # Assume cognee imported


class CogneeStore(BaseStore):
    def __init__(self, cognee_client):
        self.client = cognee_client  # Your Cognee instance
        self.graph = cg.create_graph("langgraph_memory")  # Or your graph

    async def put(self, namespace: Sequence[str | None], key: str, value: Any) -> None:
        # Serialize value (JSON/dict expected)
        full_key = self._make_key(namespace, key)
        self.graph.add_node(full_key, {"data": value, "embedding": self.embed(str(value))})

    async def get(self, namespace: Sequence[str | None], key: str) -> Optional[Any]:
        full_key = self._make_key(namespace, key)
        node = self.graph.get_node(full_key)
        return node["data"] if node else None

    async def search(
        self,
        namespace: Sequence[str | None],
        filter: Optional[dict] = None,
        query: Optional[str] = None,
        **kwargs: Any,
    ) -> list[Any]:
        # Semantic search + filter
        full_ns = self._make_key(namespace, "")
        results = self.graph.search(
            query=query or "", namespace=full_ns[:1]
        )  # Adapt to Cognee search
        if filter:
            results = [r for r in results if self._matches_filter(r["data"], filter)]
        return [r["data"] for r in results[: kwargs.get("limit", 4)]]

    async def delete(self, namespace: Sequence[str | None], key: Optional[str] = None) -> None:
        if key:
            full_key = self._make_key(namespace, key)
            self.graph.delete_node(full_key)
        else:
            # Delete namespace
            pass  # Implement bulk delete

    async def list_keys(self, namespace: Sequence[str | None]) -> list[str]:
        # Optional: list keys in namespace
        pass

    def _make_key(self, namespace: Sequence[str | None], key: str) -> str:
        return "/".join(filter(None, list(namespace) + [key]))

    def embed(self, text: str) -> list[float]:
        # Use Cognee embeddings or external
        return cg.embed(text)  # Adapt to your embedding setup
