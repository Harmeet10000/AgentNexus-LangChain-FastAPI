"""6.x reachability — ancestors, not exact names; abstract base not reported."""

import ast
import pathlib
from types import SimpleNamespace

import pytest


def _collect_raises(path: pathlib.Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    raises = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Raise)
            and isinstance(node.exc, ast.Call)
            and isinstance(node.exc.func, ast.Name)
        ):
            raises.add(node.exc.func.id)
    return raises


def _collect_excepts(path: pathlib.Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    excs = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is not None:
            if isinstance(node.type, ast.Name):
                excs.add(node.type.id)
            elif isinstance(node.type, ast.Tuple):
                for elt in node.type.elts:
                    if isinstance(elt, ast.Name):
                        excs.add(elt.id)
    return excs


def _is_reachable(raised: str, excepts: set[str], ancestor_map: dict[str, set[str]]) -> bool:
    if raised in excepts:
        return True
    return any(ancestor in excepts for ancestor in ancestor_map.get(raised, set()))


def test_reachability_over_ancestors():
    raises = set()
    excepts = set()
    for p in pathlib.Path("src").rglob("*.py"):
        if p.stat().st_size == 0:
            continue
        raises |= _collect_raises(p)
        excepts |= _collect_excepts(p)
    assert "UnregisteredTaskError" in raises, "UnregisteredTaskError should be raised"
    assert "TaskPayloadValidationError" in raises, "TaskPayloadValidationError should be raised"
    assert "CeleryError" in excepts, (
        "CeleryError should be caught (covers TaskDispatchError family)"
    )
    assert "TaskDispatchError" not in raises, "TaskDispatchError base should not be raised directly"
    from app.connections.celery import (
        TaskDispatchError,
        TaskPayloadValidationError,
        UnregisteredTaskError,
    )

    ancestor_map: dict[str, set[str]] = {}
    for cls in (UnregisteredTaskError, TaskPayloadValidationError, TaskDispatchError):
        ancestor_map[cls.__name__] = {c.__name__ for c in cls.__mro__[1:]}
    assert "UnregisteredTaskError" not in excepts
    assert "TaskPayloadValidationError" not in excepts
    assert _is_reachable("UnregisteredTaskError", excepts, ancestor_map), (
        "UnregisteredTaskError reachable via CeleryError"
    )
    assert _is_reachable("TaskPayloadValidationError", excepts, ancestor_map), (
        "TaskPayloadValidationError reachable via CeleryError"
    )
    unreachable = {
        r for r in raises if r in ancestor_map and not _is_reachable(r, excepts, ancestor_map)
    }
    assert "UnregisteredTaskError" not in unreachable
    assert "TaskPayloadValidationError" not in unreachable


def test_abstract_base_not_reported_as_unreachable():
    raises = set()
    excepts = set()
    for p in pathlib.Path("src").rglob("*.py"):
        if p.stat().st_size == 0:
            continue
        raises |= _collect_raises(p)
        excepts |= _collect_excepts(p)
    assert "TaskDispatchError" not in raises
    from app.connections.celery import (
        TaskDispatchError,
        TaskPayloadValidationError,
        UnregisteredTaskError,
    )

    ancestor_map: dict[str, set[str]] = {}
    for cls in (UnregisteredTaskError, TaskPayloadValidationError, TaskDispatchError):
        ancestor_map[cls.__name__] = {c.__name__ for c in cls.__mro__[1:]}

    def is_reachable(name: str) -> bool:
        return _is_reachable(name, excepts, ancestor_map)

    unreachable = {r for r in raises if not is_reachable(r)}
    assert "TaskDispatchError" not in unreachable, "abstract base with 0 raises must not be flagged"
    assert is_reachable("TaskDispatchError"), "TaskDispatchError reachable via CeleryError ancestor"


def test_catch_order_narrowest_first():
    text = pathlib.Path("src/app/middleware/global_exception_handler.py").read_text(
        encoding="utf-8"
    )
    api_idx = text.find("isinstance(exc, APIException)")
    req_idx = text.find("isinstance(exc, RequestValidationError)")
    star_idx = text.find("isinstance(exc, StarletteHTTPException)")
    catch_all_idx = text.find("status.HTTP_500_INTERNAL_SERVER_ERROR")
    assert api_idx < req_idx < star_idx < catch_all_idx
    # The lifespan half of this ordering lives in outcome tests below
    # (`test_cognee_setup_base_caught_not_just_subclass`): the boot registry
    # expresses narrowest-first as data (`fatal_on` checked before
    # `degrade_on`), so there is no `except` text left to pin here.


def _repo_excepts() -> set[str]:
    excepts = set()
    for p in pathlib.Path("src").rglob("*.py"):
        if p.stat().st_size == 0:
            continue
        excepts |= _collect_excepts(p)
    return excepts


def test_orphan_families_caught_by_name():
    """§6: every formerly-orphan family has a by-name catch site repo-wide.

    `CogneeSetupError` is absent on purpose: the boot registry handles it as
    data (`degrade_on`), not as an `except` clause, so the AST walk below
    cannot see it. Its coverage lives in
    `test_cognee_setup_base_caught_not_just_subclass`, which degrades it
    through the real policy instead of asserting on syntax.
    """
    excepts = _repo_excepts()
    for family in (
        "CircuitBreakerOpenError",
        "IdempotencyLockError",
        "AgentMemoryError",
        "StateSchemaVersionError",
    ):
        assert family in excepts, f"{family} must be caught by name"


def _cognee_policy():
    from app.lifecycle.lifespan import STARTUP_POLICIES

    (policy,) = [p for p in STARTUP_POLICIES if p.name == "cognee"]
    return policy


def _stub_boot_app():
    return SimpleNamespace(state=SimpleNamespace())


async def test_cognee_setup_base_caught_not_just_subclass():
    """The base CogneeSetupError degrades; only the dimension subclass hard-fails.

    Outcome version of the old narrowest-first text pin. The cognee boot
    policy carries the hard-fail class in `fatal_on` while the base degrades,
    and the runner checks fatal first — which matters precisely because the
    subclass relationship (`CogneeDimensionMismatchError` extends
    `CogneeSetupError`) would let a broad-first order swallow the hard fail.
    """
    from app.lifecycle.lifespan import _run_startup_policy
    from app.shared.langchain_layer.agents.memory.cognee_client import (
        CogneeDimensionMismatchError,
        CogneeSetupError,
    )

    policy = _cognee_policy()
    assert policy.fatal_on == (CogneeDimensionMismatchError,)
    assert issubclass(CogneeDimensionMismatchError, CogneeSetupError)

    async def _misconfigured(_app, _settings):
        msg = "placeholder connection settings"
        raise CogneeSetupError(msg)

    app = _stub_boot_app()
    await _run_startup_policy(app, object(), policy._replace(setup=_misconfigured))
    assert app.state.cognee_config is None

    async def _mismatched(_app, _settings):
        msg = "embedding width 768 != 1536"
        raise CogneeDimensionMismatchError(msg)

    with pytest.raises(CogneeDimensionMismatchError):
        await _run_startup_policy(_stub_boot_app(), object(), policy._replace(setup=_mismatched))


def test_state_schema_version_caught_at_callsite():
    """hydrate_state's raise is handled in the WS session loop, not in state.py."""
    text = pathlib.Path("src/app/features/agent_saul/service.py").read_text(encoding="utf-8")
    assert "except StateSchemaVersionError" in text


def test_breaker_and_idempotency_render_deliberate_status():
    """Request-path dispatcher renders 503/409, never the 500 catch-all."""
    text = pathlib.Path("src/app/middleware/global_exception_handler.py").read_text(
        encoding="utf-8"
    )
    breaker_idx = text.find("isinstance(exc, CircuitBreakerOpenError)")
    idem_idx = text.find("isinstance(exc, IdempotencyLockError)")
    catch_all_idx = text.find("status.HTTP_500_INTERNAL_SERVER_ERROR")
    assert breaker_idx != -1
    assert idem_idx != -1
    assert breaker_idx < catch_all_idx
    assert idem_idx < catch_all_idx
    assert "HTTP_503_SERVICE_UNAVAILABLE" in text
    assert "HTTP_409_CONFLICT" in text


def test_agent_memory_narrow_before_broad():
    """Prefetch degrades on the named family before the fail-open catch-all."""
    text = pathlib.Path("src/app/shared/langchain_layer/agents/memory/prefetch.py").read_text(
        encoding="utf-8"
    )
    assert text.find("except AgentMemoryError") < text.find(
        "except Exception as exc:  # noqa: BLE001 — fail-open read path (8.3)"
    )
