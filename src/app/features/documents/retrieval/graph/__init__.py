"""Clauses-backed legal retrieval graph."""

from app.features.documents.retrieval.reranking.reranker import (
    HostedReranker,
    Reranker,
    get_configured_reranker,
    get_shared_reranker,
)

from .engine import RetrievalEngine
from .graph import build_retrieval_graph
from .nodes import _extract_postgres_chunk_ids
from .state import (
    CachedAskEnvelope,
    Citation,
    ContextGrade,
    CoverageEnvelope,
    GeneratedAnswer,
    GroundedCitation,
    QueryPlan,
    RetrievalGraphContext,
    RetrievalState,
    RetrievedChunk,
    SourceCriteria,
    StageUsage,
    StageWarning,
)

__all__ = [
    "CachedAskEnvelope",
    "Citation",
    "ContextGrade",
    "CoverageEnvelope",
    "GeneratedAnswer",
    "GroundedCitation",
    "HostedReranker",
    "QueryPlan",
    "Reranker",
    "RetrievalEngine",
    "RetrievalGraphContext",
    "RetrievalState",
    "RetrievedChunk",
    "SourceCriteria",
    "StageUsage",
    "StageWarning",
    "_extract_postgres_chunk_ids",
    "build_retrieval_graph",
    "get_configured_reranker",
    "get_shared_reranker",
]
