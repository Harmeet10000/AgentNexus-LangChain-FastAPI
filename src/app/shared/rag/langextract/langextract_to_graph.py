# src/app/shared/document_processing/langextract_to_graph.py
from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel
import langextract as lx

from src.app.shared.rag.neo4j.client import Neo4jClient  # or Graphiti client


class GraphIngestionContext(BaseModel):
    """Narrow context for graph mapping."""
    model_config = {"frozen": True}
    document_id: str
    source_url: str
    neo4j_client: Neo4jClient  # or protocol


async def ingest_extractions_to_graph(
    annotated_docs: list[lx.data.AnnotatedDocument],
    ctx: GraphIngestionContext,
) -> int:
    """Convert grounded LangExtract output → property graph nodes/rels."""
    ingested = 0

    for doc in annotated_docs:
        for extraction in doc.extractions:
            if extraction.char_interval is None:
                continue  # Skip ungrounded (hallucinated) extractions

            # Example: PERSON, ORGANIZATION, OBLIGATION, CLAUSE, DATE, etc.
            node_label = extraction.extraction_class.upper()

            # Create node with rich properties + grounding
            node_props = {
                "name": extraction.extraction_text,
                "document_id": ctx.document_id,
                "char_start": extraction.char_interval[0],
                "char_end": extraction.char_interval[1],
                "source_url": ctx.source_url,
                **extraction.attributes,  # e.g., role="counterparty", clause_number="5.2"
            }

            # Merge node
            await ctx.neo4j_client.merge_node(node_label, node_props)

            # Extract relationships from attributes or multi-pass LangExtract
            # (Better: Do a dedicated relationship extraction pass in LangExtract prompt)

            ingested += 1

    return ingested
#     Best Practice Prompt Strategy for Graphs:

# Run two passes with LangExtract:
# Entity + attribute extraction (what you already have).
# Dedicated relationship extraction pass, where you feed back the entities and ask for typed relationships with grounding.
# Why this beats plain vector RAG for legal docs

# Legal contracts have deep interdependencies (obligations, parties, clauses referencing other clauses, temporal constraints, conditions).
# Graph queries excel at traversal (e.g., “Show all obligations linked to Party X across 50 contracts”).
# Grounding from LangExtract reduces hallucinations and enables precise citation back to source text/page.
# You get both structured traversal and semantic search.
# Ingestion Flow:
# Docling preprocessing (already planned).
# LangExtract multi-pass extraction.
# Map to graph + store embeddings of grounded text spans.
# Optional: Periodic graph summarization / community detection for higher-level insights.

# Query Layer: Use LangGraph agents that can do hybrid retrieval (vector + graph traversal) with citation back to source via char_interval.
# Scaling & Reliability:

# Always filter on char_interval is not None.
# Store the full AnnotatedDocument JSONL alongside the graph for audit.
# Use Celery workers with higher CPU/memory for the Docling + LangExtract + graph ingestion pipeline.
# Cache parsed Docling Markdown aggressively.

# This turns your legal document corpus into a true queryable knowledge asset instead of isolated extractions.
