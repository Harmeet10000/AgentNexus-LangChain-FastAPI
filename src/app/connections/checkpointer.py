"""LangGraph checkpointer connection establishment.

Thin re-export: the implementation stays in
``app.shared.langgraph_layer.checkpointer`` (its task Proofs grep that file,
and its tests monkeypatch the module object), so this module only moves the
import seam — every connection-shaped import in lifespan resolves to
``app.connections``.
"""

from __future__ import annotations

from app.shared.langgraph_layer.checkpointer import (
    CheckpointerTeardown,
    setup_langgraph_checkpointer,
    teardown_langgraph_checkpointer,
)

__all__ = [
    "CheckpointerTeardown",
    "setup_langgraph_checkpointer",
    "teardown_langgraph_checkpointer",
]
