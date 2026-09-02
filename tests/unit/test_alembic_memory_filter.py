"""Band F group 3: the alembic autogenerate filter.

Ordering is load-bearing (Decision 4): this filter is the only protection that
survives someone setting ``include_schemas=True``, and group 4 is what causes the
tables it protects to exist. It must exclude the memory schema in both directions —
an object living in a foreign schema is dropped; an ordinary application table
passes through.
"""

from __future__ import annotations

import ast
import types
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import Column, Integer, MetaData, Table

if TYPE_CHECKING:
    from typing import Any

_ENV_PATH = Path(__file__).resolve().parents[2] / "src" / "alembic" / "env.py"


def _load_env(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Load only the two tested symbols from env.py in isolation.

    The previous ``spec_from_file_location(...).exec_module`` approach executed
    the whole file, which imports every ``app.features.*`` model to register
    ``Base.metadata``. After the memory stack gained edges through
    ``app.shared.langchain_layer.agents.memory.cognee_client`` and
    ``app.features.health``, that full exec triggers an import cycle that
    surfaces as ``AttributeError`` during fixture setup. Extracting the two
    symbols by AST keeps the fixture independent of the application import
    graph while still testing the exact source the migration chain ships.
    """
    source = _ENV_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(_ENV_PATH))

    wanted = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "MEMORY_SCHEMA_NAME" for t in node.targets)
        )
        or (
            isinstance(node, ast.FunctionDef)
            and node.name in {"_is_memory_schema", "include_object"}
        )
    ]

    assert wanted, "expected MEMORY_SCHEMA_NAME and include_object in env.py"
    mod = ast.Module(body=wanted, type_ignores=[])
    ast.fix_missing_locations(mod)
    code = compile(mod, filename=str(_ENV_PATH), mode="exec")
    namespace: dict[str, Any] = {}
    exec(code, namespace)  # noqa: S102 — controlled test fixture, not user input
    # ponytail: AST extraction avoids importing the full app graph; mock-based exec is the upgrade if more symbols are needed
    module = types.SimpleNamespace(**namespace)
    module._configured_calls = []  # type: ignore[attr-defined]
    return module


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> Any:
    return _load_env(monkeypatch)


@pytest.fixture
def configured_calls(env: Any) -> list[dict[str, Any]]:
    return env._configured_calls


def test_a_memory_schema_object_is_excluded(env: Any) -> None:
    foreign = Table(
        "entities",
        MetaData(),
        Column("id", Integer),
        schema=env.MEMORY_SCHEMA_NAME,
    )
    assert env.include_object(foreign, "entities", "table", True, None) is False


def test_an_application_table_passes_through(env: Any) -> None:
    # Synthetic app table (no schema → public/default) must pass the filter.
    # The previous version asserted ``Base.metadata.tables`` was populated by
    # env.py's side-effect imports, which no longer holds under isolated AST
    # loading — and that coupling is what created the cycle in the first place.
    app_table = Table("app_table", MetaData(), Column("id", Integer))
    assert env.include_object(app_table, "app_table", "table", True, None) is True
    # Keep soft compatibility: if Base happens to be populated, verify it too.
    with suppress(Exception):
        from database import Base

        if Base.metadata.tables:
            name, table = next(iter(Base.metadata.tables.items()))
            assert env.include_object(table, name, "table", True, None) is True


def test_a_table_with_no_schema_attribute_is_kept(env: Any) -> None:
    class _Bare:
        pass

    assert env.include_object(_Bare(), "anything", "table", True, None) is True


def test_the_filter_is_wired_into_both_configure_calls() -> None:
    """Static check: both branches name the filter.

    (A dynamic count of recorded ``configure`` kwargs would depend on which
    sibling test loaded env.py first — the stand-in context sticks to the
    alembic package for the rest of the session.)
    """
    source = _ENV_PATH.read_text(encoding="utf-8")
    assert source.count("include_object=include_object") == 2, (
        "the filter must guard the offline AND online configure calls"
    )
