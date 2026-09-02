"""Every relational database catcher preserves bounded driver diagnostics."""

from __future__ import annotations

import ast
from pathlib import Path

_REPOSITORIES = (
    "audit/repository.py",
    "credits/repositories/credit_repository.py",
    "credits/repositories/consumption_repository.py",
    "documents/repository.py",
    "invoices/repository.py",
    "payments/repository.py",
    "plans/repository.py",
    "subscriptions/repository.py",
    "webhooks/repository.py",
)


def _handlers(path: Path) -> list[ast.ExceptHandler]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler)]


def _exception_names(handler: ast.ExceptHandler) -> set[str]:
    if isinstance(handler.type, ast.Name):
        return {handler.type.id}
    if isinstance(handler.type, ast.Tuple):
        return {element.id for element in handler.type.elts if isinstance(element, ast.Name)}
    return set()


def _contains_call(nodes: list[ast.stmt], attribute: str) -> bool:
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attribute
        for statement in nodes
        for node in ast.walk(statement)
    )


def test_relational_handlers_note_before_rollback_and_failure() -> None:
    root = Path("src/app/features")
    checked = 0
    for relative_path in _REPOSITORIES:
        for handler in _handlers(root / relative_path):
            names = _exception_names(handler)
            if not names & {"IntegrityError", "SQLAlchemyError"}:
                continue
            checked += 1
            body = handler.body
            note_index = next(
                index
                for index, statement in enumerate(body)
                if any(
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "add_database_error_note"
                    for node in ast.walk(statement)
                )
            )
            rollback_index = next(
                index
                for index, statement in enumerate(body)
                if _contains_call([statement], "rollback")
            )
            failure_index = next(
                index
                for index, statement in enumerate(body)
                if isinstance(statement, ast.Return)
                and isinstance(statement.value, ast.Call)
                and isinstance(statement.value.func, ast.Name)
                and statement.value.func.id == "Failure"
            )
            assert note_index < rollback_index < failure_index
            if "IntegrityError" in names:
                assert any(
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "add_database_error_note"
                    for statement in body
                    for node in ast.walk(statement)
                )
    assert checked == 69


def test_rollback_gate_fixture_has_forbidden_and_permitted_forms() -> None:
    root = Path(".ast-grep/fixtures/repository-rollback-required")
    forbidden = ast.parse((root / "forbid.py").read_text(encoding="utf-8"))
    permitted = ast.parse((root / "permit.py").read_text(encoding="utf-8"))

    def has_rollback_before_failure(tree: ast.Module) -> bool:
        for handler in ast.walk(tree):
            if not isinstance(handler, ast.ExceptHandler):
                continue
            positions = {
                "rollback": [
                    node.lineno
                    for statement in handler.body
                    for node in ast.walk(statement)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "rollback"
                ],
                "failure": [
                    node.lineno
                    for statement in handler.body
                    for node in ast.walk(statement)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "Failure"
                ],
            }
            if positions["failure"]:
                return bool(
                    positions["rollback"] and min(positions["rollback"]) < min(positions["failure"])
                )
        return True

    assert not has_rollback_before_failure(forbidden)
    assert has_rollback_before_failure(permitted)
