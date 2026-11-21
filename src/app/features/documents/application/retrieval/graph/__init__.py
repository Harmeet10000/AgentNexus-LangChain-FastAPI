"""Clauses-backed legal retrieval graph."""

from typing import TYPE_CHECKING

from app.features.documents.application.retrieval.reranking.reranker import (
    HostedReranker,
    Reranker,
    get_configured_reranker,
    get_shared_reranker,
)

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

if TYPE_CHECKING:
    from .engine import RetrievalEngine
    from .graph import build_retrieval_graph
    from .nodes import _extract_postgres_chunk_ids

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


def __getattr__(name: str) -> object:
    """Lazy engine/graph/nodes re-exports that break the service import cycle.

    ``retrieval.service`` imports ``row_hydration`` and ``graph.state``; both
    resolve through this package, and a module-level ``.nodes`` import here
    would pull ``graph.nodes`` — which imports ``_is_graphiti_branch_failure``
    from the still-partially-initialized ``retrieval.service`` — whenever
    ``retrieval.service`` is the first module imported in a fresh process
    (ImportError). The state and reranking names have no cycle, so they stay
    eager; the engine/graph-builder/node helpers load on first attribute
    access instead.
    """
    # Every lazy import below carries a PLC0415 noqa — the late import is the
    # cycle break itself (see the docstring).
    if name == "RetrievalEngine":
        from .engine import RetrievalEngine  # noqa: PLC0415

        return RetrievalEngine
    if name == "build_retrieval_graph":
        from .graph import build_retrieval_graph  # noqa: PLC0415

        return build_retrieval_graph
    if name == "_extract_postgres_chunk_ids":
        from .nodes import _extract_postgres_chunk_ids  # noqa: PLC0415

        return _extract_postgres_chunk_ids
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
