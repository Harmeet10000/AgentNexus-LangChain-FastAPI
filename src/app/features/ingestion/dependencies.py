from typing import Annotated, Any

from fastapi import Depends, Request
from langgraph.graph.state import CompiledStateGraph

from app.features.auth import CurrentClaims
from app.utils import ServiceUnavailableException


async def get_ingestion_graph(request: Request) -> CompiledStateGraph[Any]:
    graph = request.app.state.ingestion_graph
    if graph is None:
        msg = "Ingestion graph is not wired"
        raise ServiceUnavailableException(msg)
    return graph


async def get_current_user_id(claims: CurrentClaims) -> str:
    return claims.sub


IngestionGraphDep = Annotated[CompiledStateGraph[Any], Depends(get_ingestion_graph)]
UserIdDep = Annotated[str, Depends(get_current_user_id)]
