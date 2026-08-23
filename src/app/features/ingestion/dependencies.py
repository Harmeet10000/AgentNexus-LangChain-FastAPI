from typing import Annotated, Any

from fastapi import Depends, Request
from langgraph.graph.state import CompiledStateGraph

from app.features.auth import CurrentClaims
from app.utils import ServiceUnavailableException


async def get_ingestion_graph(request: Request) -> CompiledStateGraph[Any]:
    """The compiled ingestion graph, or a typed 503 if this process has none.

    **This dependency is expected to fail, permanently.** The lifespan assignment that would
    populate it is commented out by decision (D17) and stays that way: ingestion runs in the queue
    worker process, which never executes the application lifespan, so a per-process graph was never
    shared application state to begin with. The router that would consume this is not mounted
    either, so no unavailable surface actually ships today. What this function owes is that if it is
    ever reached, it fails *closed* and says why.

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
