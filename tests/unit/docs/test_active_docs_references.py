from __future__ import annotations

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DOCS_ROOT = REPOSITORY_ROOT / "docs-site"


def _active_docs_and_tooling() -> list[Path]:
    """Return active docs/tooling files, excluding archived or historical content."""
    return [
        *sorted(DOCS_ROOT.glob("*.md")),
        *sorted(DOCS_ROOT.glob("*.txt")),
        *sorted(DOCS_ROOT.rglob("*.mdx")),
        *sorted((DOCS_ROOT / "scripts").glob("*.py")),
    ]


def test_active_docs_and_tooling_use_the_current_navigation_file() -> None:
    offenders = [
        path.relative_to(REPOSITORY_ROOT).as_posix()
        for path in _active_docs_and_tooling()
        if "mint.json" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []
