from typing import Annotated, Any

from fastapi import Depends, Request
from langgraph.graph.state import CompiledStateGraph

from app.features.auth.api.http.dependencies import CurrentClaims
from app.utils import ServiceUnavailableException


async def get_ingestion_graph(request: Request) -> CompiledStateGraph[Any]:
    """The compiled ingestion graph, or a typed 503 if this process has none.

    **This dependency is expected to fail in the serving process.** The lifespan DOES provision
    ``app.state.ingestion_graph`` (``lifecycle/lifespan.py`` → ``provide_document_ingestion_graph``),
    but that graph is bound for the queue-worker contract; in the API process the attribute is
    absent, so this dependency fails closed with a typed 503. The deprecated router that consumes it
    is retained only for the sunset window (see the quarantine ledger's folded-shim row). What this
    function owes is exactly what it does: if reached without a provisioned graph, it fails *closed*
    and says why — never a 500, never a fallback answering path.

    Read through ``getattr`` with a default rather than as an attribute, because the attribute is
    never **set** — not set to ``None``. Starlette's state object raises on an unknown attribute, so
    a direct read raised before the ``is None`` test below could run, and the guard that looks like
    it produces a 503 produced an unhandled attribute error and a 500 instead. Both cases now reach
    the same branch: never provisioned, and explicitly provisioned as absent.
    """
    graph = getattr(request.app.state, "ingestion_graph", None)
    if graph is None:
        msg = "Ingestion graph is not provisioned in this process"
        raise ServiceUnavailableException(msg, data={"capability": "ingestion_graph"})
    return graph


async def get_current_user_id(claims: CurrentClaims) -> str:
    return claims.sub


IngestionGraphDep = Annotated[CompiledStateGraph[Any], Depends(get_ingestion_graph)]
UserIdDep = Annotated[str, Depends(get_current_user_id)]
