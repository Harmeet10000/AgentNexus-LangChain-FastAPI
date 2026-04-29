"""State and schemas for clauses-backed legal retrieval."""

from __future__ import annotations

from typing import Literal  # noqa: TC003 - Pydantic resolves Literal at model build time.

from langgraph.graph import MessagesState
from pydantic import BaseModel, ConfigDict, Field


class QueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rewritten_query: str
    sub_queries: list[str] = Field(default_factory=list)
    route: Literal["hybrid_postgres", "graph_neo4j", "both"] = "hybrid_postgres"
    query_type: Literal["exact", "conceptual", "mixed"] = "mixed"
    vector_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    keyword_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    jurisdiction: str | None = None
    contract_type: str | None = None
    bm25_threshold: float | None = None
    exact_phrase: str | None = None


class RetrievedChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    chunk_text: str
    preamble: str
    clause_type: str
    parent_doc_id: str
    metadata_: dict[str, object]
    custom_metadata: dict[str, object]
    score: float


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    clause_type: str
    claim: str


class ContextGrade(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sufficient: bool
    missing_aspects: list[str] = Field(default_factory=list)
    rewrite_suggestion: str | None = None


class GeneratedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: Literal["high", "medium", "uncertain"] = "uncertain"


class RetrievalState(MessagesState, total=False):
    user_id: str
    query: str
    doc_ids_filter: list[str]
    query_plan: QueryPlan
    graph_chunk_ids: list[str]
    retrieved_chunks: list[RetrievedChunk]
    reranked_chunks: list[RetrievedChunk]
    context_grade: ContextGrade
    iteration_count: int
    generated_answer: GeneratedAnswer
    cache_hit: bool
    cached_answer: GeneratedAnswer
