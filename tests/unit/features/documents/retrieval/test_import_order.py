"""Import-order regression protection for the retrieval package (wave 3).

Greptile finding: importing ``retrieval.service`` first in a fresh process
raised ``ImportError`` — the package ``__init__`` pulled ``graph.nodes``, which
imports ``_is_graphiti_branch_failure`` from the still-partially-initialized
``retrieval.service``. The tests hid it by importing
``app.features.documents.application.queries.documents`` first. The lazy
engine/graph/node re-exports in the package ``__init__`` break the cycle; these
tests pin every entry point in its own interpreter.
"""

from __future__ import annotations

import subprocess  # noqa: S404 — a fresh interpreter is the only faithful import-order probe
import sys

import pytest

pytestmark = pytest.mark.unit

_ENTRY_POINTS = (
    "import app.features.documents.retrieval.service",
    "from app.features.documents.retrieval.graph import RetrievalEngine",
    "from app.features.documents.retrieval.graph import build_retrieval_graph",
    "from app.features.documents.retrieval.graph.nodes import make_hybrid_retrieval_node",
    "import app.features.documents.retrieval.graph.state",
    "import app.features.documents.retrieval.row_hydration",
)


@pytest.mark.parametrize("entry_point", _ENTRY_POINTS)
def test_each_entry_point_imports_in_a_fresh_process(entry_point: str) -> None:
    result = subprocess.run(  # noqa: S603 — fixed argv, no shell
        [sys.executable, "-c", entry_point],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, (
        f"{entry_point!r} failed in a fresh process:\n{result.stderr[-2000:]}"
    )
    assert "circular import" not in result.stderr.lower()
