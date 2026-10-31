"""Migration 0034 source pins: idempotent against both database shapes.

Migrations cannot run hermetically, so the contract is pinned in source — the
existing style for worker commands. Two shapes reach 0034:

- 0004-executed databases: search tables carry the full retrieval index set.
- 0014-repaired databases (stamped, never migrated): ``0014`` recreates the
  tables with **no indexes at all** (its docstring records that
  ``search_chunks_bm25_idx`` was referenced by settings but never created).

Therefore every drop must be ``IF EXISTS``-guarded and every retrieval index
the search branches query through must be created ``IF NOT EXISTS`` with the
exact name/definition the ORM declares — a drifted index name or method would
make search silently match nothing after one of the two paths.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
_MIGRATION = (_REPO_ROOT / "src/alembic/versions/0034_search_tenant_scoping.py").read_text(
    encoding="utf-8"
)
_MODELS = (_REPO_ROOT / "src/app/features/search/adapters/persistence/models.py").read_text(
    encoding="utf-8"
)


def _class_tablename(node: ast.ClassDef) -> str | None:
    """The literal ``__tablename__`` of an ORM class, if declared."""
    for stmt in node.body:
        if (
            isinstance(stmt, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "__tablename__"
                for target in stmt.targets
            )
            and isinstance(stmt.value, ast.Constant)
        ):
            return str(stmt.value.value)
    return None


def _orm_index_names(table: str) -> set[str]:
    """Index names declared on the ``table`` ORM class only.

    Scoped to the ClassDef whose ``__tablename__`` matches, so an Index in
    another model (present or future) can neither satisfy nor break this pin.
    """
    tree = ast.parse(_MODELS)
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or _class_tablename(node) != table:
            continue
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Call) or ast.unparse(inner.func) != "Index":
                continue
            if not inner.args or not isinstance(inner.args[0], ast.Constant):
                continue
            names.add(str(inner.args[0].value))
    return names


def test_every_index_drop_is_if_exists_guarded() -> None:
    """The 0014-repaired shape has no indexes; a bare drop would abort."""
    assert "DROP INDEX IF EXISTS ix_search_chunks_content_tsv_gin" in _MIGRATION
    assert (
        "op.drop_index"
        not in _MIGRATION.split("def upgrade()", 1)[1].split("def downgrade()", 1)[0]
    )


def test_retrieval_indexes_are_created_if_not_exists() -> None:
    """The four retrieval indexes plus the FK btree, exactly as 0004 defined them."""
    assert (
        "CREATE INDEX IF NOT EXISTS search_chunks_bm25_idx ON search_chunks\n"
        "    USING bm25(content) WITH (text_config='english')" in _MIGRATION
    )
    assert (
        "CREATE INDEX IF NOT EXISTS search_chunks_embedding_idx ON search_chunks\n"
        "    USING diskann (embedding vector_cosine_ops)" in _MIGRATION
    )
    assert (
        "CREATE INDEX IF NOT EXISTS ix_search_chunks_content_trgm ON search_chunks\n"
        "    USING gin (content gin_trgm_ops)" in _MIGRATION
    )
    assert (
        "CREATE INDEX IF NOT EXISTS ix_search_chunks_chunk_metadata_gin ON search_chunks\n"
        "    USING gin (chunk_metadata)" in _MIGRATION
    )
    assert (
        "CREATE INDEX IF NOT EXISTS ix_search_chunks_document_id ON search_chunks (document_id)"
        in _MIGRATION
    )


def test_migration_creates_every_index_the_orm_declares() -> None:
    """ORM-declared index names must all appear in the migration source.

    An index present in the registry but absent from the database is one
    ``alembic check`` proposes to REMOVE — and the bm25 index name is passed
    to ``to_bm25query()`` as a literal, so a silent drop makes keyword
    retrieval match nothing without raising.
    """
    for index_name in _orm_index_names("search_chunks"):
        assert index_name in _MIGRATION, f"ORM index {index_name} missing from 0034"


def test_constraint_and_column_drops_are_if_exists_guarded() -> None:
    """Consistent with the dual-shape hardening (CR MAJOR): no bare drops."""
    upgrade_section = _MIGRATION.split("def upgrade()", 1)[1].split("def downgrade()", 1)[0]
    assert "DROP CONSTRAINT IF EXISTS uq_search_documents_content_hash" in upgrade_section
    assert "DROP COLUMN IF EXISTS content_tsv" in upgrade_section
    assert "op.drop_constraint" not in upgrade_section
    assert "op.drop_column" not in upgrade_section


def test_composite_ownership_foreign_key_is_created() -> None:
    """Chunks must reference a matching (id, tenant_id) pair on the parent."""
    assert "uq_search_documents_id_tenant_id" in _MIGRATION
    assert "fk_search_chunks_document_tenant" in _MIGRATION
    assert "FOREIGN KEY (document_id, tenant_id)" in _MIGRATION
    assert "REFERENCES search_documents (id, tenant_id)" in _MIGRATION


def test_downgrade_drops_are_if_exists_guarded() -> None:
    downgrade_section = _MIGRATION.split("def downgrade()", 1)[1]
    assert "DROP CONSTRAINT IF EXISTS fk_search_chunks_document_tenant" in downgrade_section
    assert "DROP CONSTRAINT IF EXISTS uq_search_documents_tenant_content_hash" in downgrade_section
    assert "DROP COLUMN IF EXISTS tenant_id" in downgrade_section
    assert "DROP COLUMN IF EXISTS status" in downgrade_section


def test_extensions_are_created_if_not_exists() -> None:
    """The 0014 shape never ran 0004's extension setup either."""
    for extension in ("vector", "vectorscale", "pg_textsearch", "pg_trgm", "unaccent"):
        assert f"CREATE EXTENSION IF NOT EXISTS {extension}" in _MIGRATION


def test_downgrade_restores_the_tsv_index_with_gin() -> None:
    """0004's tsv index was GIN; a btree recreation would not serve it (G P2)."""
    assert (
        "CREATE INDEX ix_search_chunks_content_tsv_gin ON search_chunks USING gin (content_tsv)"
        in _MIGRATION
    )


def test_downgrade_preflights_cross_tenant_duplicate_hashes() -> None:
    """The global unique constraint cannot survive cross-tenant dedupe (G P1)."""
    assert "COUNT(DISTINCT tenant_id) > 1" in _MIGRATION
    assert "this downgrade never deletes tenant data automatically" in _MIGRATION
