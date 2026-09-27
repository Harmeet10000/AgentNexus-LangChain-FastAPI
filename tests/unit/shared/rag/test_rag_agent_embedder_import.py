"""Architectural guards for the retired Docling-specific embedder."""

import ast
from importlib.util import find_spec
from pathlib import Path

_EXAMPLE = Path(__file__).resolve().parents[4] / "src" / "app" / "examples" / "rag_agent_advanced.py"


def test_docling_specific_embedder_module_is_retired() -> None:
    assert find_spec("app.shared.rag.docling.embedder") is None


def test_example_query_calls_use_the_shared_embedding_entry_point() -> None:
    source = _EXAMPLE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "embed_text"
    ]

    assert "app.shared.rag.docling.embedder" not in source
    assert len(calls) == 5
