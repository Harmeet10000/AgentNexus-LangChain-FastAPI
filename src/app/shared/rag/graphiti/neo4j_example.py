# Example 2: Using Neo4j in endpoints with dependency injection
# ────────────────────────────────────────────────────────────

from fastapi import APIRouter, Depends
from neo4j import AsyncDriver
from pydantic import BaseModel

from app.connections import get_neo4j_driver, get_neo4j_session
from app.utils.logger import logger

router = APIRouter(prefix="/api/v1/graph", tags=["Graph DB"])


class NodeData(BaseModel):
    """Node data model."""

    labels: list[str]
    properties: dict


@router.get("/nodes")
async def get_nodes(
    driver: AsyncDriver = Depends(get_neo4j_driver),
) -> dict:
    """Get first 10 nodes from Neo4j.

    Example:
        GET /api/v1/graph/nodes
    """
    try:
        async with get_neo4j_session(driver) as session:
            result = await session.run(
                """
                MATCH (n)
                RETURN labels(n) as labels, properties(n) as properties
                LIMIT 10
                """
            )
            records = await result.all()

            logger.info(f"Retrieved {len(records)} nodes")

            return {
                "total": len(records),
                "nodes": [
                    {
                        "labels": record.get("labels"),
                        "properties": record.get("properties"),
                    }
                    for record in records
                ],
            }

    except Exception as e:
        logger.error(f"Failed to get nodes: {e!s}", exc_info=True)
        raise


@router.post("/create-node")
async def create_node(
    label: str, properties: dict, driver: AsyncDriver = Depends(get_neo4j_driver)
) -> dict:
    """Create a new node in Neo4j.

    Example:
        POST /api/v1/graph/create-node?label=Person
        {
            "name": "John Doe",
            "age": 30,
            "city": "New York"
        }
    """
    try:
        async with get_neo4j_session(driver) as session:
            # Build properties string for Cypher query
            props_str = ", ".join(f"{k}: ${k}" for k in properties.keys())

            query = f"""
            CREATE (n:{label} {{{props_str}}})
            RETURN n, id(n) as node_id
            """

            result = await session.run(query, **properties)
            record = await result.single()

            logger.info(f"Created node with label {label}")

            return {
                "success": True,
                "node_id": record.get("node_id") if record else None,
                "label": label,
                "properties": properties,
            }

    except Exception as e:
        logger.error(f"Failed to create node: {e!s}", exc_info=True)
        raise


@router.get("/search/{query}")
async def search_cypher(
    query: str, driver: AsyncDriver = Depends(get_neo4j_driver)
) -> dict:
    """Execute a custom Cypher query.

    ⚠️ WARNING: Only use with validated/trusted input in production!

    Example:
        GET /api/v1/graph/search/MATCH%20(n)%20RETURN%20n%20LIMIT%205
    """
    try:
        async with get_neo4j_session(driver) as session:
            result = await session.run(query)
            records = await result.all()

            logger.info(f"Cypher query returned {len(records)} records")

            return {
                "query": query,
                "total": len(records),
                "records": [dict(record) for record in records],
            }

    except Exception as e:
        logger.error(f"Cypher query failed: {e!s}", exc_info=True)
        raise


@router.get("/relationships/{node_id}")
async def get_relationships(
    node_id: int, driver: AsyncDriver = Depends(get_neo4j_driver)
) -> dict:
    """Get relationships for a specific node.

    Example:
        GET /api/v1/graph/relationships/123
    """
    try:
        async with get_neo4j_session(driver) as session:
            result = await session.run(
                """
                MATCH (n)-[r]-(m)
                WHERE id(n) = $node_id
                RETURN type(r) as relationship_type, m, r
                LIMIT 50
                """,
                node_id=node_id,
            )
            records = await result.all()

            logger.info(f"Retrieved {len(records)} relationships for node {node_id}")

            return {
                "node_id": node_id,
                "total_relationships": len(records),
                "relationships": [
                    {
                        "type": record.get("relationship_type"),
                        "target_node": dict(record.get("m")),
                    }
                    for record in records
                ],
            }

    except Exception as e:
        logger.error(
            f"Failed to get relationships for node {node_id}: {e!s}",
            exc_info=True,
        )
        raise


# Example 3: Using Neo4j with LangChain Neo4jGraph
# ────────────────────────────────────────────────────

from langchain_neo4j import Neo4jGraph

from app.config import get_settings


async def get_langchain_graph() -> Neo4jGraph:
    """Initialize LangChain's Neo4jGraph wrapper.

    Use this for RAG and knowledge graph operations with LangChain.
    """
    settings = get_settings()

    graph = Neo4jGraph(
        url=settings.NEO4J_URI,
        username=settings.NEO4J_USERNAME,
        password=settings.NEO4J_PASSWORD,
        database=settings.NEO4J_DATABASE,
        enhanced_schema=True,
    )

    logger.info("LangChain Neo4jGraph initialized")
    return graph


@router.post("/rag/query")
async def rag_query(question: str) -> dict:
    """Execute RAG query using LangChain Neo4jGraph.

    Example:
        POST /api/v1/graph/rag/query
        {
            "question": "Who are the main characters?"
        }
    """
    try:
        graph = await get_langchain_graph()

        # Example: Get schema
        schema = graph.get_schema

        # Example: Query knowledge graph
        result = await graph.query(
            f"MATCH (n:Character) WHERE n.role CONTAINS '{question}' RETURN n LIMIT 5"
        )

        logger.info(f"RAG query executed: {question}")

        return {
            "question": question,
            "schema": schema,
            "result": result,
        }

    except Exception as e:
        logger.error(f"RAG query failed: {e!s}", exc_info=True)
        raise
