"""Process-scoped retrieval-engine provisioning (`compiled-graph-lifecycle`).

The compiled retrieval graph is a first-class served surface (mount-retrieval-graph
tasks 3.1 / 4.x): it compiles at most once per process, and every request-scoped
collaborator — the repository over a short-lived session, the cache client, the
relationship-graph client, the language model, the optional grader seam — is
built per invocation. Nothing request-scoped is captured at compile time, and a
provisioning failure at lifespan degrades to ``None`` in the serving state (the
startup policy registry owns that), where the ask endpoints and the qna engine
fail closed with a typed unavailability.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from app.features.documents.application.retrieval.graph.graph import build_retrieval_graph
from app.features.documents.application.retrieval.graph.state import RetrievalGraphContext

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable
    from typing import Any

    from langgraph.graph.state import CompiledStateGraph

    from app.features.documents.application.retrieval.reranking.reranker import Reranker


class RetrievalEngine:
    """The compiled retrieval graph plus its per-invocation collaborator seam.

    ``graph`` is the compile-once surface both serving doors share (the HTTP
    ask endpoints and Agent Saul's qna node). ``invocation()`` yields the
    compiled graph together with a freshly built :class:`RetrievalGraphContext`
    — the repository's session opens and closes around exactly one graph
    invocation, so no session is pinned across a request or a graph run.
    """

    def __init__(
        self,
        *,
        session_factory: Any,
        redis: Any | None,
        graphiti: Any | None,
        llm_factory: Callable[[], Any],
        grader: Any | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._redis = redis
        self._graphiti = graphiti
        self._llm_factory = llm_factory
        self._grader = grader
        self._reranker = reranker
        # Compile once, at provisioning time — never per request.
        self.graph: CompiledStateGraph[Any, RetrievalGraphContext] = build_retrieval_graph()

    @asynccontextmanager
    async def invocation(
        self,
    ) -> AsyncIterator[
        tuple[
            CompiledStateGraph[Any, RetrievalGraphContext],
            RetrievalGraphContext,
        ]
    ]:
        """One graph invocation's compiled graph and request-scoped context."""
        # Local import (noqa: PLC0415): the engine composes the persistence
        # adapter at invocation time, keeping the graph package free of it.
        from app.connections.postgres import independent_session  # noqa: PLC0415
        from app.features.documents.adapters.persistence.postgres.document_repository import (  # noqa: PLC0415
            DocumentRepository,
        )

        async with independent_session(self._session_factory) as session:
            from app.features.documents.application.retrieval.graph.budgets import (  # noqa: PLC0415
                build_stage_budget_tracker,
            )

            context = RetrievalGraphContext(
                llm=self._llm_factory(),
                repo=DocumentRepository(session),
                redis=self._redis,
                graphiti=self._graphiti,
                reranker=self._reranker,
                grader=self._grader,
                # Per-ask stage budgets (G6 task 4.1); None when the feature is
                # disabled in settings.
                budgets=build_stage_budget_tracker(),
            )
            yield self.graph, context
