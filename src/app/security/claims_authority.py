"""Static guard helpers for the JWT claims authority policy."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

_CLAIM_NAMES = {"claims", "token_claims", "jwt_claims", "raw_claims"}
_PRIVILEGE_FIELDS = {"role", "permissions"}
_CLAIM_SERIALIZERS = {"dict", "model_dump", "model_dump_json"}


def _is_claim_expression(node: ast.expr, aliases: set[str] | None = None) -> bool:
    if isinstance(node, ast.Name):
        name = node.id.lower()
        return name in _CLAIM_NAMES or name in (aliases or set()) or "claim" in name
    if isinstance(node, ast.Attribute):
        return node.attr.lower() in _CLAIM_NAMES or "claim" in node.attr.lower()
    return False


def find_claim_privilege_violations(source_root: Path) -> list[str]:  # noqa: PLR0912
    """Return direct JWT role/permission reads in application source."""
    violations: list[str] = []
    app_root = source_root / "src" / "app"
    for path in sorted(app_root.rglob("*.py")):
        # This module owns JWT wire serialization; it is the one intentional
        # place where role/permission claim keys are encoded and decoded.
        if path.relative_to(app_root).as_posix() == "security/tokens.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        claim_aliases = set(_CLAIM_NAMES)
        aliases_changed = True
        while aliases_changed:
            aliases_changed = False
            for assignment in ast.walk(tree):
                if not isinstance(assignment, (ast.Assign, ast.AnnAssign)):
                    continue
                value = assignment.value
                if value is None or not _is_claim_expression(value, claim_aliases):
                    continue
                targets = (
                    assignment.targets
                    if isinstance(assignment, ast.Assign)
                    else [assignment.target]
                )
                for target in targets:
                    if isinstance(target, ast.Name) and target.id not in claim_aliases:
                        claim_aliases.add(target.id)
                        aliases_changed = True
        for node in ast.walk(tree):
            field: str | None = None
            base: ast.expr | None = None
            if isinstance(node, ast.Attribute) and node.attr in _PRIVILEGE_FIELDS:
                field, base = node.attr, node.value
            elif isinstance(node, ast.Subscript):
                key = node.slice
                if isinstance(key, ast.Constant) and key.value in _PRIVILEGE_FIELDS:
                    field, base = str(key.value), node.value
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if (
                    node.func.attr == "get"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value in _PRIVILEGE_FIELDS
                    and _is_claim_expression(node.func.value, claim_aliases)
                ):
                    field, base = str(node.args[0].value), node.func.value
                elif node.func.attr in _CLAIM_SERIALIZERS and _is_claim_expression(
                    node.func.value, claim_aliases
                ):
                    field, base = "serialization", node.func.value
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "dict"
                and node.args
                and _is_claim_expression(node.args[0], claim_aliases)
            ):
                field, base = "serialization", node.args[0]
            if field is not None and base is not None and _is_claim_expression(base, claim_aliases):
                line_no = getattr(node, "lineno", 0)
                violations.append(f"{path}:{line_no}: claims.{field} is not authority")
    return violations
