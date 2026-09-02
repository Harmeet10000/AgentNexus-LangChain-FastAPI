"""6.x reachability — ancestors, not exact names; abstract base not reported."""

import ast
import pathlib


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
    text2 = pathlib.Path("src/app/lifecycle/lifespan.py").read_text(encoding="utf-8")
    assert text2.find("CogneeDimensionMismatchError") < text2.find(
        "except Exception as exc:  # noqa: BLE001 — optional dependency"
    )
