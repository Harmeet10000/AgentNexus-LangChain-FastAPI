"""Band F group 4: cognee configuration, and the startup posture around it.

The memory library is faked at its config surface (the real one writes global
state); what these tests pin is the ORDER and the REFUSALS, not the library.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from app.shared.langchain_layer.agents.memory.cognee_client import (
    _ACCESS_CONTROL_ENV_KEY,
    CogneeDimensionMismatchError,
    CogneeSetupConfig,
    CogneeSetupError,
    setup_cognee,
)

if TYPE_CHECKING:
    from typing import Any

_REAL_FIELDS = SimpleNamespace(
    host="real-db.example.com",
    port=5432,
    username="app",
    password=SecretStr("secret"),
    database="appdb",
)


def _settings(**overrides: Any) -> Any:
    base: dict[str, Any] = {
        "GEMINI_FLASH_MODEL": "gemini-flash",
        "GEMINI_EMBEDDING_MODEL": "gemini-embedding",
        "GEMINI_API_KEY": SecretStr("key"),
        "EMBEDDING_DIMENSION": 768,
        "NEO4J_URI": "bolt://neo4j:7687",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": SecretStr("pass"),
        "POSTGRES_HOST": "real-db.example.com",
        "POSTGRES_PORT": 5432,
        "POSTGRES_DB_NAME": "appdb",
        "COGNEE_VECTOR_PROVIDER": "pgvector",
        "COGNEE_DB_SCHEMA": "cognee_memory",
        "COGNEE_ACCESS_CONTROL_ENABLED": True,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture(autouse=True)
def real_database_fields(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Point the accessor at a stable non-placeholder identity for every test.

    Without this the real accessor reads the environment's managed-instance URL,
    which no fake settings object can agree with.
    """
    import app.shared.langchain_layer.agents.memory.cognee_client as client

    monkeypatch.setattr(client, "get_database_fields", lambda: _REAL_FIELDS)
    return _REAL_FIELDS


@pytest.fixture
def fake_cognee(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    """Record every set_*_config call in order."""
    calls: list[tuple[str, dict[str, Any]]] = []

    class _FakeConfig:
        @staticmethod
        def _record(name: str, config_dict: dict[str, Any]) -> None:
            calls.append((name, config_dict))

        def set_llm_config(self, *, config_dict: dict[str, Any]) -> None:
            self._record("llm", config_dict)

        def set_embedding_config(self, *, config_dict: dict[str, Any]) -> None:
            self._record("embedding", config_dict)

        def set_graph_db_config(self, config_dict: dict[str, Any]) -> None:
            self._record("graph", config_dict)

        def set_relational_db_config(self, config_dict: dict[str, Any]) -> None:
            self._record("relational", config_dict)

        def set_vector_db_config(self, config_dict: dict[str, Any]) -> None:
            self._record("vector", config_dict)

    import app.shared.langchain_layer.agents.memory.cognee_client as client

    monkeypatch.setattr(client, "cognee", type("_FakeCogneeModule", (), {"config": _FakeConfig()}))
    return calls


async def test_access_control_env_is_written_before_the_first_config_call(
    fake_cognee: list[tuple[str, dict[str, Any]]],
) -> None:
    os.environ.pop(_ACCESS_CONTROL_ENV_KEY, None)
    with patch.dict(os.environ):
        await setup_cognee(_settings())
        assert os.environ[_ACCESS_CONTROL_ENV_KEY] == "true"
        assert len(fake_cognee) == 5


async def test_embedding_dimension_mismatch_raises_at_startup(
    fake_cognee: list[tuple[str, dict[str, Any]]],
) -> None:
    settings = _settings(EMBEDDING_DIMENSION=1536)
    with pytest.raises(CogneeDimensionMismatchError):
        await setup_cognee(settings)
    assert fake_cognee == [], "the refusal must precede any library call"


async def test_divergent_postgres_host_is_a_named_configuration_error(
    fake_cognee: list[tuple[str, dict[str, Any]]],
) -> None:
    with pytest.raises(CogneeSetupError, match="disagree"):
        await setup_cognee(_settings(POSTGRES_HOST="other-db.example.com"))


async def test_placeholder_connection_fields_are_refused(
    fake_cognee: list[tuple[str, dict[str, Any]]],
) -> None:
    import app.shared.langchain_layer.agents.memory.cognee_client as client

    client.get_database_fields = lambda: SimpleNamespace(  # type: ignore[assignment]
        host="localhost", port=5432, username="u", password="p", database="db"
    )
    settings = _settings(POSTGRES_HOST="localhost", POSTGRES_DB_NAME="db")
    with pytest.raises(CogneeSetupError, match="placeholder"):
        await setup_cognee(settings)
    assert fake_cognee == [], "no library call may happen after a refused configuration"


async def test_the_setup_result_is_typed_and_carries_no_credential(
    fake_cognee: list[tuple[str, dict[str, Any]]],
) -> None:
    result = await setup_cognee(_settings())
    assert isinstance(result, CogneeSetupConfig)
    dumped = result.model_dump().keys()
    for forbidden in ("password", "api_key", "url", "secret", "token"):
        assert not any(forbidden in field.lower() for field in dumped), forbidden
    assert result.vector_provider == "pgvector"
    assert result.embedding_dimension == 768
