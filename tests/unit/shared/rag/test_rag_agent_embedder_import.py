"""Unit tests for task A4 — the phantom ``ingestion.embedder`` import is gone.

A4's own second Proof asks that ``rag_agent_advanced`` import cleanly, and it
cannot: that module has **nine undefined names** across four identifiers
(``RunContext`` ×6, ``Agent``, ``List``, ``itemgetter``), and ``pydantic-ai`` —
which supplies ``Agent`` and ``RunContext`` — is declared in neither
``pyproject.toml`` nor ``uv.lock``. Because the module carries no
``from __future__ import annotations``, those annotations are evaluated when each
``def`` executes, so the failure is at import, not at call. None of that is A4's
doing: ``git show HEAD`` has the same six ``RunContext`` occurrences, and A4's
diff is one added import against four removed ones.

So these tests prove what A4 actually protects, statically:

* the phantom module reference is gone and cannot come back unnoticed;
* the surviving import sits at **module level**, so a future breakage is an
  import error rather than a first-call error — which is the property A4's
  requirement is really about;
* the import *target* resolves and exposes both members the former call sites
  invoke, so retargeting did not merely trade one deferred failure for another.

The last test pins the blocker itself. It is written to fail the moment the
module becomes importable, which is deliberate: leg E owns this file (Q-A
decided it relocates to ``src/app/examples/``), and whoever does that work should
be forced to revisit A4's Proof rather than discover it stale.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
from pathlib import Path

# Deliberately NOT imported. Every fact below is read from the source text,
# because importing this module is the one thing that does not work — see
# `test_module_remains_unimportable_pending_leg_e`.
_TARGET = "app.shared.rag.document_processing.embedder"
_SOURCE_PATH = (
    Path(__file__).resolve().parents[4] / "src" / "app" / "shared" / "rag" / "rag_agent_advanced.py"
)


def _source() -> str:
    return _SOURCE_PATH.read_text(encoding="utf-8")


def _tree() -> ast.Module:
    return ast.parse(source=_source(), filename=str(object=_SOURCE_PATH))


def _embedder_imports(tree: ast.Module) -> list[ast.ImportFrom]:
    return [
        node
        for node in ast.walk(node=tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and "embedder" in node.module
    ]


def test_source_file_is_where_this_test_expects() -> None:
    """Guards the path arithmetic above, so a later failure is never a typo."""
    assert _SOURCE_PATH.is_file()


# --- The phantom is gone ---


def test_no_reference_to_the_phantom_module_remains() -> None:
    assert "ingestion.embedder" not in _source()


def test_the_phantom_package_does_not_exist() -> None:
    """``src/`` top-level packages are alembic, app, database, lynk, mcp_core, tasks.

    Asserted so that a regression fails on this test rather than on a
    ``ModuleNotFoundError`` at some caller's first request.
    """
    assert importlib.util.find_spec("ingestion") is None


# --- Exactly one embedder import, at module level ---


def test_there_is_exactly_one_embedder_import() -> None:
    """Four function-local imports collapsed to one. The count is the point.

    Four copies of the same import is how the phantom survived: each was inside a
    different function, so no single call path exercised more than one of them.
    """
    assert len(_embedder_imports(tree=_tree())) == 1


def test_the_embedder_import_is_at_module_level() -> None:
    tree = _tree()
    module_level = [node for node in tree.body if isinstance(node, ast.ImportFrom)]
    assert _embedder_imports(tree=tree)[0] in module_level


def test_the_embedder_import_targets_the_surviving_module() -> None:
    node = _embedder_imports(tree=_tree())[0]
    assert node.module == _TARGET
    assert [alias.name for alias in node.names] == ["create_embedder"]


# --- The target resolves, and exposes what the former call sites call ---


def test_the_import_target_resolves() -> None:
    assert importlib.util.find_spec(_TARGET) is not None


def test_the_embedder_exposes_every_member_the_call_sites_invoke() -> None:
    """``embed_query`` is the one that mattered.

    The four former call sites call ``embedder.embed_query(...)``. Retargeting the
    import without that member would have traded ``ModuleNotFoundError`` for
    ``AttributeError`` — a different exception at the same moment, which is not
    what A4 asks for.
    """
    embedder = importlib.import_module(name=_TARGET).create_embedder()
    assert callable(embedder.embed_query)
    assert callable(embedder.embed_chunks)


def test_every_call_site_uses_a_member_the_embedder_provides() -> None:
    """Derived from the source rather than listed, so a new call site is covered."""
    embedder = importlib.import_module(name=_TARGET).create_embedder()
    called = {
        node.func.attr
        for node in ast.walk(node=_tree())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "embedder"
    }
    assert called
    assert called <= {name for name in dir(embedder) if not name.startswith("_")}


# --- The blocker A4 does not mention, pinned so leg E must revisit it ---


def test_module_remains_unimportable_pending_leg_e() -> None:
    """A4's second Proof is unsatisfiable, and not because of A4.

    ``Agent`` and ``RunContext`` come from ``pydantic-ai``, which is not a declared
    dependency. Adding one to make a CLI that nothing imports importable is a
    dependency decision A4 never asked for, and leg E may well delete or rewrite
    this file instead — Q-A already decided it relocates to ``src/app/examples/``.

    When that happens this test fails, which is the intent: it is a tripwire, not
    an endorsement.
    """
    assert importlib.util.find_spec("pydantic_ai") is None

    tree = _tree()
    imported = {
        alias.asname or alias.name.split(sep=".")[0]
        for node in ast.walk(node=tree)
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in node.names
    }
    used = {node.id for node in ast.walk(node=tree) if isinstance(node, ast.Name)}

    assert {"Agent", "RunContext"} <= used
    assert not ({"Agent", "RunContext"} & imported)
