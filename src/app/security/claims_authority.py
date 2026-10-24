"""Static guard helpers for the JWT claims authority policy."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

_CLAIM_NAMES = {"claims", "token_claims", "jwt_claims", "raw_claims"}
_PRIVILEGE_FIELDS = {"role", "permissions"}


def _is_claim_expression(node: ast.expr) -> bool:
    if isinstance(node, ast.Name):
        name = node.id.lower()
        return name in _CLAIM_NAMES or "claim" in name
    if isinstance(node, ast.Attribute):
        return node.attr.lower() in _CLAIM_NAMES or "claim" in node.attr.lower()
    return False


def find_claim_privilege_violations(source_root: Path) -> list[str]:
    """Return direct JWT role/permission reads in application source."""
    violations: list[str] = []
    app_root = source_root / "src" / "app"
    for path in sorted(app_root.rglob("*.py")):
        # This module owns JWT wire serialization; it is the one intentional
        # place where role/permission claim keys are encoded and decoded.
        if path.relative_to(app_root).as_posix() == "security/tokens.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            field: str | None = None
            base: ast.expr | None = None
            if isinstance(node, ast.Attribute) and node.attr in _PRIVILEGE_FIELDS:
                field, base = node.attr, node.value
            elif isinstance(node, ast.Subscript):
                key = node.slice
                if isinstance(key, ast.Constant) and key.value in _PRIVILEGE_FIELDS:
                    field, base = str(key.value), node.value
            if field is not None and base is not None and _is_claim_expression(base):
                line_no = getattr(node, "lineno", 0)
                violations.append(f"{path}:{line_no}: claims.{field} is not authority")
    return violations
