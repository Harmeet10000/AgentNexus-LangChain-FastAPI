"""Knowledge Graph RAG - Graphiti retrieval composed with LangChain generation."""

from __future__ import annotations

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

from app.shared.langchain_layer.models import build_chat_model

_GRAPHITI = Graphiti("neo4j://localhost:7687", "neo4j", "password")
_CHAT = build_chat_model()


async def ingest_document(text: str, source: str) -> None:
    """Ingest a source document as a Graphiti episode."""
    await _GRAPHITI.add_episode(
        name=source,
        episode_body=text,
        source=EpisodeType.text,
        source_description=f"Document: {source}",
    )


async def search_knowledge_graph(query: str, *, top_k: int = 5) -> str:
    """Hybrid Graphiti search response summarized by LangChain chat model."""
    results = await _GRAPHITI.search(query=query, num_results=top_k)
    context_parts = [
        f"Entity: {item.node.name}\n"
        f"Type: {item.node.type}\n"
        f"Context: {item.context}\n"
        f"Relationships: {item.relationships}"
        for item in results
    ]
    context = "\n---\n".join(context_parts) if context_parts else "No graph context found."
    prompt = f"Answer the query using graph context.\n\nQuery: {query}\n\nContext:\n{context}"
    response = await _CHAT.ainvoke(prompt)
    return str(response.content)
