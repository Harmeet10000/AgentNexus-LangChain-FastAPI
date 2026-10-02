"""Ingestion delegation conformance (agent-message-standard 4.1).

The ingestion graph moves between stages via graph edges and `Send`, never via
agent delegation: no node binds tools, invokes a sub-agent, or emits tool calls,
and `IngestionState` carries no message channel. These guards pin that shape —
a future stage handoff MUST go through the shared delegation constructor
instead of inventing its own pair format.
"""

import ast
from pathlib import Path

_NODES = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "app"
    / "shared"
    / "langgraph_layer"
    / "ingestion_kb"
    / "nodes.py"
).read_text(encoding="utf-8")
_STATE = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "app"
    / "shared"
    / "langgraph_layer"
    / "ingestion_kb"
    / "state.py"
).read_text(encoding="utf-8")


def test_stage_nodes_emit_no_tool_calls() -> None:
    tree = ast.parse(_NODES)
    names = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert "bind_tools" not in names
    assert "transfer_to_orchestrator" not in _NODES
    assert "ToolMessage" not in _NODES


def test_ingestion_state_carries_no_message_channel() -> None:
    assert "messages" not in _STATE
