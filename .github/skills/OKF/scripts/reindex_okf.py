#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["PyYAML>=6,<7", "markdown-it-py>=3,<5", "mdit-py-plugins>=0.4,<1"]
# ///
"""Regenerate marked indexes, graph, and catalog while retaining manual indexes."""

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import quote

from validate_okf import RESERVED, authored_body, frontmatter, links, local_target

MARKER = "<!-- okf:generated-index -->"


def documents(root) -> dict:
    records = {}
    for path in sorted(root.rglob("*.md")):
        if path.name in RESERVED or path.is_symlink():
            continue
        meta, body = frontmatter(path.read_text())
        if meta is None:
            message = f"Missing frontmatter: {path}"
            raise ValueError(message)
        records[path] = (meta, body)
    return records


def directories(root, records) -> set[Path]:
    result = {root}
    for path in records:
        parent = path.parent
        while parent.is_relative_to(root):
            result.add(parent)
            if parent == root:
                break
            parent = parent.parent
    return result


def index_content(root, directory, dirs, records) -> str:
    body = MARKER + "\n\n# " + directory.name.replace("-", " ").title() + "\n\n"
    if directory == root:
        body = '---\nokf_version: "0.2"\n---\n\n' + body
    for child in sorted(d for d in dirs if d.parent == directory and d != directory):
        body += f"* [{child.name.replace('-', ' ').title()}]({quote(child.name)}/index.md) - Browse this collection.\n"
    for path, (meta, _) in sorted(records.items()):
        if path.parent != directory:
            continue
        title = str(meta.get("title", path.stem)).replace("[", "").replace("]", "")
        description = str(meta.get("description", "Knowledge document.")).replace("\n", " ")
        body += f"* [{title}]({quote(path.name)}) - {description}\n"
    return body


def outgoing(root, path, meta, body, records) -> set[tuple[str, str, str]]:
    key = path.relative_to(root).as_posix().removesuffix(".md")
    authored, _ = authored_body(body)
    result = set()
    for href in links(authored):
        target = local_target(root, path, href)
        if target in records:
            result.add((key, target.relative_to(root).as_posix().removesuffix(".md"), "links-to"))
    for source in meta.get("sources", []):
        value = source.get("resource", "")
        if value.startswith(("/", "./", "../")):
            target = local_target(root, path, value)
            if target in records:
                result.add(
                    (key, target.relative_to(root).as_posix().removesuffix(".md"), "derived-from")
                )
    return result


def link_graph(root, records) -> dict:
    nodes = []
    edges = set()
    for path, (meta, body) in sorted(records.items()):
        key = path.relative_to(root).as_posix()
        nodes.append(
            {
                "id": key.removesuffix(".md"),
                "path": key,
                "title": meta.get("title", path.stem),
                "type": meta["type"],
                "tags": meta.get("tags", []),
            }
        )
        edges.update(outgoing(root, path, meta, body, records))
    return {
        "format": "okf-link-graph/1",
        "nodes": nodes,
        "edges": [{"source": a, "target": b, "relation": c} for a, b, c in sorted(edges)],
    }


def rebuild(root, dry_run=False) -> dict:
    root = Path(root).resolve()
    if not root.is_dir():
        message = f"Bundle directory does not exist: {root}"
        raise ValueError(message)
    records = documents(root)
    dirs = directories(root, records)
    graph = link_graph(root, records)
    catalog = {
        path.relative_to(root).as_posix(): meta for path, (meta, _) in sorted(records.items())
    }
    catalog_json = json.dumps(catalog, ensure_ascii=False, indent=2, default=str)
    rewritten = 0
    for directory in sorted(dirs):
        index = directory / "index.md"
        if index.exists() and MARKER not in index.read_text():
            continue
        if not dry_run:
            index.write_text(index_content(root, directory, dirs, records))
        rewritten += 1
    destination = root / "maintenance"
    if not dry_run:
        destination.mkdir(exist_ok=True)
        (destination / "graph.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2))
        (destination / "catalog.json").write_text(catalog_json)
    return {"indexes": rewritten, "documents": len(graph["nodes"]), "links": len(graph["edges"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Example: uv run reindex_okf.py knowledge --dry-run. "
        "Exit codes: 0 success; 2 invalid bundle or I/O. Validate before and after rebuilding.",
    )
    parser.add_argument("bundle", type=Path, help="Actual OKF bundle root")
    parser.add_argument(
        "--dry-run", action="store_true", help="Report planned counts without writing"
    )
    args = parser.parse_args()
    try:
        sys.stdout.write(json.dumps(rebuild(args.bundle, dry_run=args.dry_run)) + "\n")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
