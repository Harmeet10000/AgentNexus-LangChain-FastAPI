"""Static gate: every index or constraint name in source must be created by a migration.

The hole this closes is narrow and was expensive. Query text named ``clauses_bm25_idx``; a
migration created that index; the table it was created on was never created by any migration. So
every check that asked "does a migration create this index?" answered yes, and the index still did
not exist anywhere. A gate that only compares *names* passes the exact defect it was built for.

Three rules, and the third is the one that matters:

1. A name that appears in source and that no migration creates is reported.
2. A name that a migration creates on a table that migration history also creates is **not**
   reported — that is the healthy case.
3. A name that a migration creates on a table no migration creates **is** reported, even though
   the ``CREATE INDEX`` exists.

**This module opens no connection and imports no driver, engine or session factory.** It is
standard library only — not merely as hygiene but because the requirement is that the failure must
not depend on a reachable database. A gate that could be satisfied by pointing at a live instance
would go green the moment someone applied the missing table by hand, which is the situation that
produced the hole. It also means the module can be imported and pointed at a checkout of another
revision, which is what makes red-before/green-after provable rather than asserted.

Both entry points take paths and return data:

- ``audit(source_root, migrations_root) -> list[Finding]`` for the pytest guard.
- ``python -m app.utils.schema_identifier_gate <source_root> <migrations_root>`` for a tree that
  is not this one. Exit status is ``0`` with no findings and ``1`` otherwise.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

# What counts as an index or constraint name. Two shapes cover every identifier this project
# names in query text: a conventional prefix, or the `_idx` suffix.
#
# `_index` and `_key` are deliberately NOT suffix triggers, and leaving them out is the difference
# between a usable gate and one nobody runs. `chunk_index` is a column and appears in almost every
# query in the retrieval path; `idempotency_key` is a column and appears as a mapping key in the
# outbox code. Either as a trigger would bury the real findings under dozens of false ones. The
# one identifier here that ends in `_index` — `uq_chunks_document_chunk_index` — is caught by its
# `uq_` prefix, so nothing is lost.
#
# Case-sensitive on purpose: identifiers in this repository are lowercase, and folding case would
# start matching Python constant names such as the ones in `features/documents/constants.py`
# without adding a single real detection.
_IDENTIFIER = re.compile(r"\b(?:(?:uq|ux|ix|pk|fk|ck)_[a-z0-9_]+|[a-z0-9_]+_idx)\b")

_CREATE_TABLE_SQL = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[\"']?([a-z0-9_]+)",
    re.IGNORECASE,
)
_CREATE_TABLE_OP = re.compile(r"op\.create_table\(\s*[\"']([a-z0-9_]+)[\"']")

_CREATE_INDEX_SQL = re.compile(
    r"CREATE\s+(?:UNIQUE\s+)?INDEX\s+(?:CONCURRENTLY\s+)?(?:IF\s+NOT\s+EXISTS\s+)?"
    r"[\"']?([a-z0-9_]+)[\"']?\s+ON\s+(?:ONLY\s+)?[\"']?([a-z0-9_]+)",
    re.IGNORECASE,
)
_CREATE_INDEX_OP = re.compile(
    r"op\.create_index\(\s*(?:index_name\s*=\s*)?[\"']([a-z0-9_]+)[\"']\s*,\s*"
    r"(?:table_name\s*=\s*)?[\"']([a-z0-9_]+)[\"']"
)

_ADD_CONSTRAINT_SQL = re.compile(
    r"ALTER\s+TABLE\s+(?:ONLY\s+)?[\"']?([a-z0-9_]+)[\"']?\s+ADD\s+CONSTRAINT\s+[\"']?([a-z0-9_]+)",
    re.IGNORECASE,
)
_CREATE_CONSTRAINT_OP = re.compile(
    r"op\.create_(?:unique|check|foreign_key|primary_key)_constraint\(\s*"
    r"(?:constraint_name\s*=\s*)?[\"']([a-z0-9_]+)[\"']\s*,\s*"
    r"(?:(?:table_name|source_table)\s*=\s*)?[\"']([a-z0-9_]+)[\"']"
)

# Constraints declared inline in a table definition carry a `name=` and no table of their own.
# They are attributed to the nearest *preceding* `create_table` in the same file — which is where
# autogenerate puts them, and the heuristic is stated here rather than hidden because a
# constraint declared outside any table block would be attributed to the wrong table by it.
_INLINE_CONSTRAINT = re.compile(
    r"name\s*=\s*[\"']((?:uq|ux|ix|pk|fk|ck)_[a-z0-9_]+|[a-z0-9_]+_idx)[\"']"
)

MISSING_CREATOR = "MISSING_CREATOR"
ORPHANED_TABLE = "ORPHANED_TABLE"
UNPARSED_SOURCE = "UNPARSED_SOURCE"

# Implicit string concatenation is how long DDL actually gets written, and it defeats every
# pattern above. `0004_subscriptions_allow_resubscribe.py` reads:
#
#     op.execute(
#         f"CREATE UNIQUE INDEX uq_subscriptions_user_plan_active "
#         f"ON subscriptions (user_id, plan_id) "
#
# so between the index name and its `ON` clause the *file text* holds a closing quote, a newline,
# indentation, a string prefix and an opening quote. `\s+` cannot cross that, the creation is
# missed, and the gate reports a live index as uncreated. The first run of this gate produced
# exactly that false positive, which is worth knowing: a gate's own defects read as findings, so a
# new report is a claim to verify against migration history before it is a claim about the schema.
#
# Collapsing the glue reassembles the statement. A comma between two literals is not whitespace,
# so genuinely separate strings in a list or call are left alone.
_LITERAL_GLUE = re.compile(r"[\"']\s*[rbfRBF]{0,2}[\"']")


@dataclass(frozen=True)
class Reference:
    """One occurrence of an identifier name inside a string literal in source."""

    name: str
    path: str
    line: int

    def where(self) -> str:
        return f"{self.path}:{self.line}"


@dataclass(frozen=True)
class Creation:
    """One identifier that migration history creates, and the table it lands on."""

    name: str
    table: str | None
    path: str


@dataclass(frozen=True)
class Finding:
    """A reported defect. ``locations`` are source sites, not migration sites."""

    kind: str
    name: str
    detail: str
    locations: tuple[str, ...]

    def render(self) -> str:
        sites = " ".join(self.locations) or "-"
        return f"{self.kind}\t{self.name}\t{self.detail}\t{sites}"


def _docstring_ids(tree: ast.Module) -> set[int]:
    """Identify string constants that are docstrings.

    Prose is excluded from the scan. A docstring that names a phantom index is misleading to a
    reader, but it is not query text, and a gate that fails on explanatory writing teaches people
    to stop writing it.
    """
    ids: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        body = node.body
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            ids.add(id(first.value))
    return ids


def _python_files(root: Path, *, excluded: tuple[Path, ...]) -> Iterator[Path]:
    for path in sorted(root.rglob("*.py")):
        if any(path.is_relative_to(skip) for skip in excluded):
            continue
        if "__pycache__" in path.parts:
            continue
        yield path


def scan_source(
    source_root: Path, *, excluded: tuple[Path, ...] = ()
) -> tuple[list[Reference], list[str]]:
    """Collect identifier references from string literals under ``source_root``.

    Returns the references and the paths that could not be parsed. An unreadable file is returned
    rather than skipped: a file the gate cannot see is a hole in the gate, and reporting a clean
    result over a partial scan is the failure mode this whole change keeps running into.
    """
    references: list[Reference] = []
    unparsed: list[str] = []
    for path in _python_files(source_root, excluded=excluded):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError, OSError):
            unparsed.append(str(path))
            continue
        skip = _docstring_ids(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if id(node) in skip:
                continue
            references.extend(
                Reference(name=match.group(0), path=str(path), line=node.lineno)
                for match in _IDENTIFIER.finditer(node.value)
            )
    return references, unparsed


def scan_migrations(migrations_root: Path) -> tuple[dict[str, Creation], set[str]]:
    """Collect the identifiers and tables that migration history creates.

    Read as text rather than parsed, because a revision mixes ``op.*`` calls with raw statements
    inside ``op.execute`` and both forms create real objects. An AST walk would have to reach
    inside the string literals anyway.
    """
    creations: dict[str, Creation] = {}
    tables: set[str] = set()
    if not migrations_root.exists():
        return creations, tables

    for path in sorted(migrations_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            text = _LITERAL_GLUE.sub("", path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, OSError):
            continue

        table_offsets: list[tuple[int, str]] = []
        for pattern in (_CREATE_TABLE_SQL, _CREATE_TABLE_OP):
            for match in pattern.finditer(text):
                tables.add(match.group(1))
                table_offsets.append((match.start(), match.group(1)))
        table_offsets.sort()

        for pattern in (_CREATE_INDEX_SQL, _CREATE_INDEX_OP, _CREATE_CONSTRAINT_OP):
            for match in pattern.finditer(text):
                name, table = match.group(1), match.group(2)
                creations.setdefault(name, Creation(name=name, table=table, path=str(path)))
        for match in _ADD_CONSTRAINT_SQL.finditer(text):
            # Argument order is reversed in this one: the table comes first.
            table, name = match.group(1), match.group(2)
            creations.setdefault(name, Creation(name=name, table=table, path=str(path)))
        for match in _INLINE_CONSTRAINT.finditer(text):
            name = match.group(1)
            enclosing = [table for offset, table in table_offsets if offset < match.start()]
            creations.setdefault(
                name,
                Creation(name=name, table=enclosing[-1] if enclosing else None, path=str(path)),
            )
    return creations, tables


def audit(source_root: Path, migrations_root: Path) -> list[Finding]:
    """Report every source-referenced identifier that migration history does not really create."""
    source_root = source_root.resolve()
    migrations_root = migrations_root.resolve()
    # The migrations root normally sits *inside* the source root. Scanning it as source would
    # read every `CREATE INDEX` body as a reference and make the gate assert against itself.
    excluded = (migrations_root,) if migrations_root.is_relative_to(source_root) else ()

    references, unparsed = scan_source(source_root, excluded=excluded)
    creations, tables = scan_migrations(migrations_root)

    sites: dict[str, list[str]] = {}
    for reference in references:
        sites.setdefault(reference.name, []).append(reference.where())

    findings: list[Finding] = [
        Finding(kind=UNPARSED_SOURCE, name=path, detail="file could not be parsed", locations=())
        for path in unparsed
    ]
    for name in sorted(sites):
        locations = tuple(dict.fromkeys(sites[name]))
        creation = creations.get(name)
        if creation is None:
            findings.append(
                Finding(
                    kind=MISSING_CREATOR,
                    name=name,
                    detail="no migration creates it",
                    locations=locations,
                )
            )
        elif creation.table is not None and creation.table not in tables:
            findings.append(
                Finding(
                    kind=ORPHANED_TABLE,
                    name=name,
                    detail=f"created on '{creation.table}', which no migration creates",
                    locations=locations,
                )
            )
    return findings


def main(argv: Sequence[str] | None = None) -> int:
    """Run the gate over two roots and report. Returns ``1`` when anything is found."""
    parser = argparse.ArgumentParser(
        prog="python -m app.utils.schema_identifier_gate",
        description="Report index and constraint names in source that no migration truly creates.",
    )
    parser.add_argument("source_root", type=Path)
    parser.add_argument("migrations_root", type=Path)
    args = parser.parse_args(argv)

    findings = audit(args.source_root, args.migrations_root)
    for finding in findings:
        sys.stdout.write(f"{finding.render()}\n")
    count = len(findings)
    sys.stdout.write(f"{count} finding(s)\n")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
