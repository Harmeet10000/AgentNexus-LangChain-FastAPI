"""Band F group 3: the alembic autogenerate filter.

Ordering is load-bearing (Decision 4): this filter is the only protection that
survives someone setting ``include_schemas=True``, and group 4 is what causes the
tables it protects to exist. It must exclude the memory schema in both directions —
an object living in a foreign schema is dropped; an ordinary application table
passes through.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Column, Integer, MetaData, Table

_ENV_PATH = Path(__file__).resolve().parents[2] / "src" / "alembic" / "env.py"


def _load_env(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Load env.py by path — it is a script, not a package module.

    ``alembic.context`` only works under an alembic runner, so a recording stand-in
    is swapped in for the import; the offline branch it triggers at import time is
    harmless by construction and its ``configure`` kwargs are captured.
    """
    configured: list[dict[str, Any]] = []

    class _FakeConfig:
        config_file_name = None

        @staticmethod
        def get_main_option(_name: str) -> str | None:
            return None

    fake_context = types.SimpleNamespace(
        config=_FakeConfig(),
        is_offline_mode=lambda: True,
        configure=lambda **kwargs: configured.append(kwargs),
        begin_transaction=lambda: nullcontext(),
        run_migrations=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "alembic.context", fake_context)

    spec = importlib.util.spec_from_file_location("_alembic_env", _ENV_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._configured_calls = configured
    return module


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> Any:
    return _load_env(monkeypatch)


@pytest.fixture()
def configured_calls(env: Any) -> list[dict[str, Any]]:
    return env._configured_calls


def test_a_memory_schema_object_is_excluded(env: Any) -> None:
    foreign = Table(
        "entities",
        MetaData(),
        Column("id", Integer),
        schema=env.MEMORY_SCHEMA_NAME,
    )
    assert env.exclude_non_app_schema(foreign, "entities", "table") is False


def test_an_application_table_passes_through(env: Any) -> None:
    from database import Base

    assert Base.metadata.tables, "the registry must be populated by env.py's imports"
    name, table = next(iter(Base.metadata.tables.items()))
    assert env.exclude_non_app_schema(table, name, "table") is True


def test_a_table_with_no_schema_attribute_is_kept(env: Any) -> None:
    class _Bare:
        pass

    assert env.exclude_non_app_schema(_Bare(), "anything", "table") is True


def test_the_filter_is_wired_into_both_configure_calls() -> None:
    """Static check: both branches name the filter.

    (A dynamic count of recorded ``configure`` kwargs would depend on which
    sibling test loaded env.py first — the stand-in context sticks to the
    alembic package for the rest of the session.)
    """
    source = _ENV_PATH.read_text(encoding="utf-8")
    assert source.count("include_object=exclude_non_app_schema") == 2, (
        "the filter must guard the offline AND online configure calls"
    )
