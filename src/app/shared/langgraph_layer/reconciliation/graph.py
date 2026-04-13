from __future__ import annotations

from typing import TYPE_CHECKING, cast

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .pipeline_node import (
    make_apply_changes_node,
    make_fetch_existing_node,
    make_reconcile_node,
    make_write_versions_node,
)
from .state import ReconciliationState

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from .state import ReconciliationRunnable


def build_reconciliation_graph(
    reconcile_llm: ReconciliationRunnable,
    db_engine: AsyncEngine,
) -> CompiledStateGraph:
    """Build the reconciliation graph once during application startup."""
    graph = StateGraph(ReconciliationState)
    graph.add_node("fetch_existing", cast("object", make_fetch_existing_node(db_engine)))
    graph.add_node("reconcile", cast("object", make_reconcile_node(reconcile_llm)))
    graph.add_node("apply_changes", cast("object", make_apply_changes_node(db_engine)))
    graph.add_node("write_versions", cast("object", make_write_versions_node(db_engine)))

    graph.set_entry_point("fetch_existing")
    graph.add_edge("fetch_existing", "reconcile")
    graph.add_edge("reconcile", "apply_changes")
    graph.add_edge("apply_changes", "write_versions")
    graph.add_edge("write_versions", END)

    return graph.compile()

