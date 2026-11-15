"""Crawler package dependency-direction contracts."""

import ast
from pathlib import Path

ROOT = Path(__file__).parents[4]
CRAWLER = ROOT / "src/app/features/crawler"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }


def test_inner_crawler_layers_do_not_import_http_transport() -> None:
    sources = [
        *sorted((CRAWLER / "application").rglob("*.py")),
        *sorted((CRAWLER / "adapters").rglob("*.py")),
        ROOT / "src/app/worker/jobs/crawler.py",
    ]

    offenders = {
        str(path.relative_to(ROOT)): sorted(
            module for module in _imports(path) if module.startswith("app.features.crawler.api")
        )
        for path in sources
        if any(module.startswith("app.features.crawler.api") for module in _imports(path))
    }

    assert offenders == {}
