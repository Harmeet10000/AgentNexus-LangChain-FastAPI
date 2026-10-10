"""Pin intentional package surfaces (restructure-domain-boundaries task 3.5).

`features/auth` and `features/billing/invoices` initializers are docstring-only:
any re-export reintroduces the eager HTTP-stack import the migration removed.
(`features/billing/credits` keeps pure dto/model type exports by design and is
not pinned here.)
"""

from __future__ import annotations

import ast
from pathlib import Path

_PINNED = (
    "src/app/features/auth/__init__.py",
    "src/app/features/billing/invoices/__init__.py",
)


def test_pinned_initializers_contain_no_imports() -> None:
    root = Path(__file__).resolve().parents[2]
    for relative in _PINNED:
        tree = ast.parse((root / relative).read_text(encoding="utf-8"))
        imports = [
            node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        assert not imports, f"{relative} re-exports runtime modules again"
