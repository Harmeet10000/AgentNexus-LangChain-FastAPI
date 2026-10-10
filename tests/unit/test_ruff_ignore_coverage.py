"""Pin per-file-ignore coverage (restructure-domain-boundaries task 2.3).

Every glob in `[tool.ruff.lint.per-file-ignores]` must match at least one
file, or be a documented intentionally-future pattern. A stale entry fails
silently in Ruff and hides the fact that the file it was written for is gone.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

# Intentionally-future patterns: kept despite matching nothing today, each
# with the rationale recorded here rather than implied.
_DOCUMENTED_FUTURE_PATTERNS = {
    "**/generated/**.py": "generated code has no annotations by policy",
}


def test_every_per_file_ignore_matches_or_is_documented() -> None:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    with pyproject.open("rb") as handle:
        config = tomllib.load(handle)
    ignores = config["tool"]["ruff"]["lint"]["per-file-ignores"]

    root = pyproject.parent
    undocumented = [
        pattern
        for pattern in ignores
        if pattern not in _DOCUMENTED_FUTURE_PATTERNS and not list(root.glob(pattern))
    ]
    assert not undocumented, f"per-file-ignores match no file: {undocumented}"
