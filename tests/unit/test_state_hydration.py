"""Band: agent-tools-unification 9.4 — state hydration (D-5).

Persisted agent state is version-checked before any reasoning step reads it:
matching version proceeds, the recognised legacy shape upgrades
deterministically, and unknown versions are refused with a typed error naming
both numbers. One constant governs writing and reading.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.shared.langgraph_layer.agent_saul import state as state_module
from app.shared.langgraph_layer.agent_saul.state import (
    STATE_SCHEMA_VERSION,
    StateSchemaVersionError,
    hydrate_state,
)


def test_matching_version_passes_through_unchanged() -> None:
    state = {"schema_version": STATE_SCHEMA_VERSION, "user_query": "q"}
    assert hydrate_state(state) is state


def test_legacy_version_zero_is_upgraded_with_missing_keys() -> None:
    legacy = {"schema_version": 0, "user_id": "u1", "status": "completed"}
    hydrated = hydrate_state(legacy)
    assert hydrated["schema_version"] == STATE_SCHEMA_VERSION
    assert hydrated["plan"] == []
    assert hydrated["working_memory"] == {}
    assert hydrated["permissions"] == {}
    assert hydrated["retry_count"] == 0
    assert hydrated["long_term_refs"] == []
    assert hydrated["user_id"] == "u1", "existing values must win over defaults"
    assert hydrated["status"] == "completed"
    assert legacy.keys() == {"schema_version", "user_id", "status"}, "input not mutated"


def test_absent_version_key_treated_as_legacy_and_upgraded() -> None:
    hydrated = hydrate_state({"user_id": "u1"})
    assert hydrated["schema_version"] == STATE_SCHEMA_VERSION
    assert hydrated["messages"] == []
    assert hydrated["qna_confidence"] == 0


def test_unknown_version_refused_naming_both_versions() -> None:
    with pytest.raises(StateSchemaVersionError) as exc_info:
        hydrate_state({"schema_version": 99})
    message = str(exc_info.value)
    assert "99" in message
    assert str(STATE_SCHEMA_VERSION) in message


def test_newer_version_is_refused_not_ignored() -> None:
    with pytest.raises(StateSchemaVersionError):
        hydrate_state({"schema_version": STATE_SCHEMA_VERSION + 1})


def test_hydrate_resolves_the_module_constant_not_an_inlined_literal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patching the module attribute changes hydrate behaviour — proof the
    function reads the imported constant at call time instead of a baked-in
    literal."""
    monkeypatch.setattr(state_module, "STATE_SCHEMA_VERSION", 424242)
    sentinel_state = {"schema_version": 424242}
    assert hydrate_state(sentinel_state) is sentinel_state
    with pytest.raises(StateSchemaVersionError) as exc_info:
        hydrate_state({"schema_version": 7})
    assert "424242" in str(exc_info.value)


def test_version_constant_has_exactly_one_definition() -> None:
    src_dir = Path(state_module.__file__).parent
    definitions = [
        path
        for path in src_dir.glob("*.py")
        if any(
            line.startswith("STATE_SCHEMA_VERSION")
            and "=" in line
            for line in path.read_text().splitlines()
        )
    ]
    assert definitions == [src_dir / "state.py"], (
        f"STATE_SCHEMA_VERSION must be defined only in state.py, found {definitions}"
    )


def test_service_layer_imports_the_constant_instead_of_a_literal() -> None:
    app_root = Path(state_module.__file__).parents[3]
    service_src = (app_root / "features/agent_saul/service.py").read_text()
    assert 'STATE_SCHEMA_VERSION' in service_src
    assert '"schema_version": 1' not in service_src
