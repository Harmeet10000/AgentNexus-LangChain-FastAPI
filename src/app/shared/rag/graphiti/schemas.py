"""
Legal domain schemas for Graphiti episodes and edges.

Design rationale:
- Episode granularity: one episode = one clause segment.
  This is the correct level because:
  * Enables temporal queries: "how did Party A's indemnity obligations
    evolve across contracts from Q1-Q3?"
  * Avoids context explosion: full-document episodes force Graphiti
    to embed too much text → poor retrieval precision
  * Enables per-clause trust scoring after human review

- Group ID = doc_id → all clause episodes from the same document
  form a named subgraph. Graphiti uses group_id to scope searches.

- EpisodeMetadata is NOT stored in the episode body (which Graphiti
  embeds as-is). It is stored in the source_description field
  as a compact JSON prefix so it's always retrievable with the episode.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from datetime import datetime
    from typing import Any


class LegalEpisodeType(StrEnum):
    """Maps to graphiti_core.nodes.EpisodeType.text for all of these.
    We use our own enum for domain clarity; convert at write time.
    """

    CLAUSE = "clause"
    FINAL_REPORT = "final_report"
    RELATIONSHIP = "relationship"


class ClauseEpisodeMetadata(BaseModel, frozen=True):
    """Stored as source_description in Graphiti. Always co-retrieved with episode."""

    doc_id: str
    clause_id: str
    clause_type: str
    jurisdiction: str
    document_type: str
    user_id: str
    thread_id: str
    human_reviewed: bool = False
    trust_score: float = Field(default=0.5, ge=0.0, le=1.0)
    schema_version: int = 1


class FinalReportEpisodeMetadata(BaseModel, frozen=True):
    doc_id: str
    user_id: str
    thread_id: str
    overall_risk_label: str
    overall_compliant: bool
    human_approved: bool
    schema_version: int = 1


class GraphitiSearchResult(BaseModel, frozen=True):
    """Normalised result returned by GraphitiService.search_*().
    Consumers never touch raw graphiti-core types.
    """

    uuid: str
    name: str
    content: str
    source_description: str
    relevance_score: float = Field(ge=0.0, le=1.0, default=0.0)
    group_id: str | None = None
    created_at: datetime | None = None
    metadata_raw: dict[str, Any] = Field(default_factory=dict)


class LegalEdgeInput(BaseModel, frozen=True):
    """Input for writing a structured legal relationship edge to Graphiti.

    Graphiti represents edges as episodes with structured body text.
    We build the body from this schema so the LLM that processes
    it gets a consistent, parseable format.
    """

    from_entity: str = Field(description="e.g. 'Acme Corp'")
    relationship: str = Field(description="e.g. 'INDEMNIFIES'")
    to_entity: str = Field(description="e.g. 'GlobalTech Ltd'")
    clause_id: str
    doc_id: str
    user_id: str
    thread_id: str
    citation_source: str
    confidence: float = Field(ge=0.0, le=1.0)

    def to_episode_body(self) -> str:
        return (
            f"{self.from_entity} {self.relationship} {self.to_entity}. "
            f"Source clause: {self.clause_id}. "
            f"Citation: {self.citation_source}. "
            f"Confidence: {self.confidence:.2f}."
        )
