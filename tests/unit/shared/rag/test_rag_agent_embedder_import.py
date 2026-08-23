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
    Path(__file__).resolve().parents[4] / "src" / "app" / "examples" / "rag_agent_advanced.py"
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


# --- The embedder import is gone; the unified entry point replaced it ---


_UNIFIED_IMPORT = "app.shared.langchain_layer.embeddings"


def test_there_is_exactly_one_unified_embeddings_import() -> None:
    """Zero embedder imports, exactly one unified-import line.

    Before band E this file asserted one ``create_embedder`` import — four
    function-local copies collapsed to module level by A4. Band E removed it
    entirely: the five call sites now use ``embed_text``, imported once.
    """
    imports = [
        node
        for node in ast.walk(node=_tree())
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and "embeddings" in node.module
    ]
    assert len(imports) == 1
    assert not _embedder_imports(tree=_tree())


def test_the_unified_import_is_at_module_level() -> None:
    tree = _tree()
    module_level = [node for node in tree.body if isinstance(node, ast.ImportFrom)]
    unified = [
        node
        for node in ast.walk(node=tree)
        if isinstance(node, ast.ImportFrom) and node.module == _UNIFIED_IMPORT
    ]
    assert unified
    assert unified[0] in module_level


def test_no_embedder_import_survives() -> None:
    """The relocation deleted the last reason this file had to touch the embedder."""
    assert _embedder_imports(tree=_tree()) == []


# --- The target resolves, and exposes what the former call sites call ---


def test_the_import_target_resolves() -> None:
    assert importlib.util.find_spec(_TARGET) is not None


def test_the_embedder_exposes_every_member_the_call_sites_invoke() -> None:
    """``embed_chunks`` is what remains after band E.

    Band E relocated ``rag_agent_advanced.py`` to ``src/app/examples/`` and
    repointed its **five** call sites (``:125``, ``:194``, ``:260``, ``:368``,
    ``:432`` — the plan and this file's earlier docstring both said four; the
    fifth is the refined-query re-search) to
    ``embed_text(..., task_type=QUERY)``, so ``embed_query`` and
    ``_Embedder.embed_query`` are deleted. The embedder keeps only what callers
    still use.
    """
    embedder = importlib.import_module(name=_TARGET).create_embedder()
    assert callable(embedder.embed_chunks)
    assert not hasattr(embedder, "embed_query")


def test_the_call_sites_reach_the_unified_query_entry_point() -> None:
    """Every query-side embedding call goes through ``embed_text(QUERY)``.

    Read from the source rather than listed, so a new call site is covered.
    """
    source = _source()
    assert (
        "from app.shared.langchain_layer.embeddings import EmbeddingTaskType, embed_text" in source
    )
    assert "embedder.embed_query" not in source
    called = sum(
        1
        for node in ast.walk(node=_tree())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "embed_text"
    )
    assert called == 5, f"expected five embed_text call sites, found {called}"


def test_every_call_site_uses_a_member_the_embedder_provides() -> None:
    """Derived from the source rather than listed, so a new call site is covered.

    After band E no ``embedder.<member>`` call remains — the attribute set is
    asserted empty rather than subset-checked.
    """
    embedder = importlib.import_module(name=_TARGET).create_embedder()
    called = {
        node.func.attr
        for node in ast.walk(node=_tree())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "embedder"
    }
    assert not called, f"stale embedder calls remain: {sorted(called)}"
    assert callable(embedder.embed_chunks)


# --- The blocker A4 does not mention, pinned so leg E must revisit it ---


def test_module_remains_unimportable_pending_leg_e() -> None:
    """A4's second Proof is unsatisfiable, and not because of A4.

    ``Agent`` and ``RunContext`` come from ``pydantic-ai``, which is not a declared
    dependency. Adding one to make a CLI that nothing imports importable is a
    dependency decision A4 never asked for. Band E executed Q-A's relocation to
    ``src/app/examples/`` and repointed the embedding call sites, but did **not**
    make the file importable — it is example code now, read not run.

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
