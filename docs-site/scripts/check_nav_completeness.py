#!/usr/bin/env python3
"""Check nav completeness for docs-site (task 6.2).

Parses docs-site/docs.json navigation, verifies:
  (a) every page path maps to an existing .mdx file
  (b) every .mdx (except 404.mdx) is referenced in nav

Exit 0 pass, 1 fail. Stdlib only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DOCS_ROOT = Path(__file__).resolve().parents[1]
MINT_JSON = DOCS_ROOT / "docs.json"


def load_nav_entries() -> list[str]:
    data = json.loads(MINT_JSON.read_text(encoding="utf-8"))
    nav = data.get("navigation", [])
    groups = nav.get("groups", []) if isinstance(nav, dict) else nav if isinstance(nav, list) else []
    entries: list[str] = []
    for g in groups:
        if not isinstance(g, dict):
            continue
        for p in g.get("pages", []):
            if isinstance(p, str):
                entries.append(p.strip())
            elif isinstance(p, dict):
                # nested group shape: {"group": "...", "pages": [...]}
                entries.extend(sub.strip() for sub in p.get("pages", []) if isinstance(sub, str))
    return entries


def main() -> int:
    if not MINT_JSON.exists():
        print(f"FAIL: {MINT_JSON} not found")
        return 1

    nav_entries = load_nav_entries()
    nav_set = set(nav_entries)

    if len(nav_entries) != len(nav_set):
        from collections import Counter

        dupes = [k for k, v in Counter(nav_entries).items() if v > 1]
        print(f"FAIL: duplicate nav entries: {dupes}")
        return 1

    mdx_files = sorted(DOCS_ROOT.rglob("*.mdx"))
    mdx_rel_map: dict[str, Path] = {}
    for p in mdx_files:
        rel = p.relative_to(DOCS_ROOT).with_suffix("").as_posix()
        mdx_rel_map[rel] = p

    errors: list[str] = []

    # (a) nav -> file
    for entry in nav_entries:
        candidate = DOCS_ROOT / f"{entry}.mdx"
        if not candidate.exists():
            errors.append(f"nav entry missing file: {entry} -> {entry}.mdx not found")

    # (b) file -> nav (except 404)
    for rel in sorted(mdx_rel_map):
        if rel == "404":
            continue
        if rel not in nav_set:
            errors.append(f"orphan mdx not in nav: {rel}.mdx")

    if errors:
        print("Nav completeness: FAIL")
        for e in errors:
            print(f"  - {e}")
        print(f"\nnav entries: {len(nav_entries)} | mdx files: {len(mdx_files)} (incl. 404.mdx)")
        return 1

    print(f"Nav completeness: PASS (nav entries: {len(nav_entries)} | mdx files: {len(mdx_files)} incl. 404.mdx)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
