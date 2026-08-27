#!/usr/bin/env python3
"""Verify Mintlify docs-site: nav↔files, frontmatter, stubs, internal links.

Checks:
  (a) every mint.json nav entry resolves to an existing .mdx file
  (b) every .mdx (except 404.mdx) is nav-listed
  (c) every file has front-matter title+description
  (d) no stub placeholder text ("TODO", "lorem", "stub") remains
  (e) internal markdown links (href="/...") resolve to a file

Exit 0 on pass, 1 on fail. Stdlib + pyyaml only.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml  # pyyaml — already in repo

DOCS_ROOT = Path(__file__).resolve().parents[1]
MINT_JSON = DOCS_ROOT / "mint.json"

# ponytail: substring scan is intentional — catches "STUB:", "TODO:" in any case.
STUB_RE = re.compile(r"\b(TODO|lorem|stub)\b", re.IGNORECASE)
LINK_RE = re.compile(r"!?\[.*?\]\(/([^)\s#?]+)([#?][^)]*)?\)")
FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def load_nav_entries() -> list[str]:
    data = json.loads(MINT_JSON.read_text(encoding="utf-8"))
    nav = data.get("navigation", {})
    groups = nav.get("groups", []) if isinstance(nav, dict) else []
    entries: list[str] = []
    for g in groups:
        if not isinstance(g, dict):
            continue
        for p in g.get("pages", []):
            # mint.json nav pages are strings; ignore dict/openapi shapes
            if isinstance(p, str):
                entries.append(p.strip())
            elif isinstance(p, dict):
                # e.g. {"group": "...", "pages": [...]} nested — flatten one level
                for sub in p.get("pages", []):
                    if isinstance(sub, str):
                        entries.append(sub.strip())
    return entries


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
    errors: list[str] = []
    warnings: list[str] = []

    # --- collect ---
    if not MINT_JSON.exists():
        print(f"FAIL: {MINT_JSON} not found")
        return 1

    nav_entries = load_nav_entries()
    nav_set = set(nav_entries)

    # duplicate nav entries
    if len(nav_entries) != len(nav_set):
        from collections import Counter

        dupes = [k for k, v in Counter(nav_entries).items() if v > 1]
        errors.append(f"duplicate nav entries: {dupes}")

    mdx_files = sorted(DOCS_ROOT.rglob("*.mdx"))
    # relative posix without .mdx suffix for comparison
    mdx_rel_map: dict[str, Path] = {}
    for p in mdx_files:
        rel = p.relative_to(DOCS_ROOT).with_suffix("").as_posix()
        mdx_rel_map[rel] = p

    # (a) nav -> file
    missing_files: list[str] = []
    for entry in nav_entries:
        candidate = DOCS_ROOT / f"{entry}.mdx"
        if not candidate.exists():
            missing_files.append(entry)
    if missing_files:
        for e in missing_files:
            errors.append(f"nav entry missing file: {e} -> {e}.mdx not found")

    # (b) file -> nav (except 404)
    orphans: list[str] = []
    for rel, path in sorted(mdx_rel_map.items()):
        if rel == "404":
            continue
        if rel not in nav_set:
            orphans.append(rel)
    if orphans:
        for o in orphans:
            errors.append(f"orphan mdx not in nav: {o}.mdx")

    # (c) frontmatter title+description
    fm_failures: list[str] = []
    for rel, path in sorted(mdx_rel_map.items()):
        text = path.read_text(encoding="utf-8")
        fm = parse_frontmatter(text)
        if fm is None:
            fm_failures.append(f"{rel}.mdx: missing or invalid frontmatter")
            continue
        title = fm.get("title")
        desc = fm.get("description")
        if not isinstance(title, str) or not title.strip():
            fm_failures.append(f"{rel}.mdx: missing/empty frontmatter 'title'")
        if not isinstance(desc, str) or not desc.strip():
            fm_failures.append(f"{rel}.mdx: missing/empty frontmatter 'description'")
    errors.extend(fm_failures)

    # (d) stub placeholders
    stub_hits: list[str] = []
    for rel, path in sorted(mdx_rel_map.items()):
        text = path.read_text(encoding="utf-8")
        # strip frontmatter so title/description containing 'stub' not flagged?
        # spec says no stub text remains anywhere — scan whole file.
        # Use word-boundary regex to avoid false positives but still catch TODO/lorem/stub.
        m = STUB_RE.search(text)
        if m:
            # report line number of first hit
            lineno = text[: m.start()].count("\n") + 1
            stub_hits.append(f"{rel}.mdx:{lineno}: contains '{m.group(0)}'")
    errors.extend(stub_hits)

    # (e) internal markdown links resolve
    link_failures: list[str] = []
    for rel, path in sorted(mdx_rel_map.items()):
        text = path.read_text(encoding="utf-8")
        for lm in LINK_RE.finditer(text):
            target = lm.group(1)  # without leading /
            # target is like "architecture/overview" or "images/logo.svg"
            # try file resolution:
            # 1) docs-site/<target>.mdx
            # 2) docs-site/<target> (e.g. images, openapi.json if ever linked)
            # 3) skip openapi.json special case — not mdx but valid
            candidate_mdx = DOCS_ROOT / f"{target}.mdx"
            candidate_file = DOCS_ROOT / target
            if candidate_mdx.exists() or candidate_file.exists():
                continue
            # also handle directory index: /guides -> guides/index.mdx
            candidate_index = DOCS_ROOT / target / "index.mdx"
            if candidate_index.exists():
                continue
            link_failures.append(f"{rel}.mdx: broken internal link '/{target}'")
    errors.extend(link_failures)

    # --- report ---
    print("Mintlify docs verification")
    print("--------------------------")
    print(f"nav entries: {len(nav_entries)} | mdx files: {len(mdx_files)} (incl. 404.mdx)")
    print(f"  (a) nav -> file: {'PASS' if not missing_files else f'FAIL ({len(missing_files)} missing)'}")
    print(f"  (b) file -> nav: {'PASS' if not orphans else f'FAIL ({len(orphans)} orphans)'}")
    print(f"  (c) frontmatter: {'PASS' if not fm_failures else f'FAIL ({len(fm_failures)} issues)'}")
    print(f"  (d) stub text:   {'PASS' if not stub_hits else f'FAIL ({len(stub_hits)} hits)'}")
    print(f"  (e) int. links:  {'PASS' if not link_failures else f'FAIL ({len(link_failures)} broken)'}")

    if errors:
        print("\nFailures:")
        for e in errors:
            print(f"  - {e}")
        if warnings:
            print("\nWarnings:")
            for w in warnings:
                print(f"  - {w}")
        print("\nOverall: FAIL")
        return 1

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  - {w}")

    print("\nOverall: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
