#!/usr/bin/env python3
"""Validate frontmatter in all docs-site .mdx files (task 6.1).

Checks every .mdx has frontmatter with non-empty ``title`` and
``description``. Reports missing/invalid per file. Exit 0 pass, 1 fail.
Stdlib + pyyaml.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

DOCS_ROOT = Path(__file__).resolve().parents[1]
FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def parse_frontmatter(text: str) -> dict | None:
    m = FRONTMATTER_RE.match(text)
    if not m:
        return None
    try:
        data = yaml.safe_load(m.group(1))
        return data if isinstance(data, dict) else None
    except yaml.YAMLError:
        return None


def main() -> int:
    mdx_files = sorted(DOCS_ROOT.rglob("*.mdx"))
    if not mdx_files:
        print("FAIL: no .mdx files found under docs-site/")
        return 1

    failures: list[str] = []
    for path in mdx_files:
        rel = path.relative_to(DOCS_ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        fm = parse_frontmatter(text)
        if fm is None:
            failures.append(f"{rel}: missing or invalid frontmatter (expected --- YAML --- block)")
            continue
        title = fm.get("title")
        desc = fm.get("description")
        if not isinstance(title, str) or not title.strip():
            failures.append(f"{rel}: missing/empty frontmatter 'title'")
        if not isinstance(desc, str) or not desc.strip():
            failures.append(f"{rel}: missing/empty frontmatter 'description'")

    if failures:
        print("Frontmatter validation: FAIL")
        for f in failures:
            print(f"  - {f}")
        print(f"\nChecked {len(mdx_files)} files — {len(failures)} issue(s)")
        return 1

    print(f"Frontmatter validation: PASS ({len(mdx_files)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
