from typing import Annotated

from fastapi import Depends, Request
from langgraph.graph.state import CompiledStateGraph


async def get_ingestion_graph(request: Request) -> CompiledStateGraph:
    return request.app.state.ingestion_graph


async def get_current_user_id(request: Request) -> str:
    return request.state.user_id


IngestionGraphDep = Annotated[CompiledStateGraph, Depends(get_ingestion_graph)]
UserIdDep = Annotated[str, Depends(get_current_user_id)]
