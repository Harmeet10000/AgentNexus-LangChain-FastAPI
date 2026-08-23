"""The identifier gate, asserted on synthetic trees.

Fixtures rather than the live tree on purpose: a test that scans `src/` passes the day it is
written and then only ever re-measures today's state. These three cases are the gate's contract,
and the third is the defect class that produced the hole — an index whose `CREATE INDEX` exists on
a table no migration creates.
"""

from __future__ import annotations

from pathlib import Path

from app.utils.schema_identifier_gate import (
    MISSING_CREATOR,
    ORPHANED_TABLE,
    UNPARSED_SOURCE,
    audit,
)


def _tree(tmp_path: Path, *, source: str, migration: str) -> tuple[Path, Path]:
    src = tmp_path / "src"
    versions = src / "alembic" / "versions"
    versions.mkdir(parents=True)
    (src / "query.py").write_text(source, encoding="utf-8")
    (versions / "0001_rev.py").write_text(migration, encoding="utf-8")
    return src, versions


def test_an_index_no_migration_creates_is_reported(tmp_path: Path) -> None:
    src, versions = _tree(
        tmp_path,
        source="SQL = \"SELECT bm25_force_merge('widgets_bm25_idx')\"\n",
        migration='def upgrade():\n    op.create_table("widgets")\n',
    )

    findings = audit(src, versions)

    assert [(f.kind, f.name) for f in findings] == [(MISSING_CREATOR, "widgets_bm25_idx")]
    assert findings[0].locations == (f"{src / 'query.py'}:1",)


def test_an_index_created_on_a_table_the_same_migration_creates_is_clean(tmp_path: Path) -> None:
    src, versions = _tree(
        tmp_path,
        source="SQL = \"SELECT bm25_force_merge('widgets_bm25_idx')\"\n",
        migration=(
            "def upgrade():\n"
            '    op.create_table("widgets")\n'
            '    op.create_index("widgets_bm25_idx", "widgets", ["body"])\n'
        ),
    )

    assert audit(src, versions) == []


def test_an_index_created_on_a_phantom_table_is_still_reported(tmp_path: Path) -> None:
    """The whole reason this gate exists. The `CREATE INDEX` is real; the table is not."""
    src, versions = _tree(
        tmp_path,
        source="SQL = \"SELECT bm25_force_merge('clauses_bm25_idx')\"\n",
        migration=(
            "def upgrade():\n"
            '    op.create_table("widgets")\n'
            '    op.execute("CREATE INDEX clauses_bm25_idx ON clauses USING bm25 (search_text)")\n'
        ),
    )

    findings = audit(src, versions)

    assert [(f.kind, f.name) for f in findings] == [(ORPHANED_TABLE, "clauses_bm25_idx")]
    assert "clauses" in findings[0].detail


def test_an_inline_unique_constraint_is_attributed_to_its_enclosing_table(tmp_path: Path) -> None:
    src, versions = _tree(
        tmp_path,
        source='SQL = "ON CONFLICT ON CONSTRAINT uq_widgets_slug DO NOTHING"\n',
        migration=(
            "def upgrade():\n"
            "    op.create_table(\n"
            '        "widgets",\n'
            '        sa.Column("slug", sa.String()),\n'
            '        sa.UniqueConstraint("slug", name="uq_widgets_slug"),\n'
            "    )\n"
        ),
    )

    assert audit(src, versions) == []


def test_column_names_that_merely_look_like_identifiers_are_not_reported(tmp_path: Path) -> None:
    """`chunk_index` and `idempotency_key` are columns, and both appear in real query text."""
    src, versions = _tree(
        tmp_path,
        source='SQL = "SELECT chunk_index, idempotency_key FROM widgets ORDER BY chunk_index"\n',
        migration='def upgrade():\n    op.create_table("widgets")\n',
    )

    assert audit(src, versions) == []


def test_a_name_in_a_docstring_is_prose_and_not_query_text(tmp_path: Path) -> None:
    src, versions = _tree(
        tmp_path,
        source='"""Historical note: widgets_bm25_idx was dropped in an earlier revision."""\n',
        migration='def upgrade():\n    op.create_table("widgets")\n',
    )

    assert audit(src, versions) == []


def test_a_file_the_gate_cannot_parse_is_reported_rather_than_skipped(tmp_path: Path) -> None:
    """A silent skip would let a scan of zero files report a clean tree."""
    src, versions = _tree(
        tmp_path,
        source="def broken(:\n",
        migration='def upgrade():\n    op.create_table("widgets")\n',
    )

    findings = audit(src, versions)

    assert [f.kind for f in findings] == [UNPARSED_SOURCE]


def test_the_migrations_root_is_not_scanned_as_source(tmp_path: Path) -> None:
    """Otherwise the `CREATE INDEX` body reads as a reference and the gate audits itself."""
    src, versions = _tree(
        tmp_path,
        source='SQL = "SELECT 1"\n',
        migration=(
            'def upgrade():\n    op.execute("CREATE INDEX ghost_bm25_idx ON ghosts (body)")\n'
        ),
    )

    assert audit(src, versions) == []


def test_a_creation_split_across_concatenated_literals_is_still_found(tmp_path: Path) -> None:
    """The gate's own first false positive, kept as a fixture.

    `0004_subscriptions_allow_resubscribe.py` writes its `CREATE UNIQUE INDEX` as adjacent
    f-string literals, so in the file text a quote, a newline, indentation and a string prefix sit
    between the index name and its `ON` clause. Without collapsing that glue the gate reports a
    live index as uncreated — a false positive indistinguishable from the real defect it hunts.
    """
    src, versions = _tree(
        tmp_path,
        source='SQL = "ON CONFLICT ON CONSTRAINT uq_widgets_slug_active DO NOTHING"\n',
        migration=(
            "def upgrade():\n"
            '    op.create_table("widgets")\n'
            "    op.execute(\n"
            '        f"CREATE UNIQUE INDEX uq_widgets_slug_active "\n'
            '        f"ON widgets (slug) "\n'
            '        f"WHERE deleted_at IS NULL"\n'
            "    )\n"
        ),
    )

    assert audit(src, versions) == []


def test_the_gate_guards_this_repository() -> None:
    """The gate over the real tree — the reason the module exists.

    The synthetic cases above pin the contract; this one points the gate at the
    checkout it lives in, so drift in this repository itself is a red test rather
    than an unrun CLI. Paths resolve from ``__file__``, not the process CWD:
    pytest can be invoked from anywhere, and a gate that silently scans an empty
    directory reports clean.
    """
    repo_root = Path(__file__).resolve().parents[2]

    findings = audit(repo_root / "src", repo_root / "src" / "alembic" / "versions")

    assert findings == []
