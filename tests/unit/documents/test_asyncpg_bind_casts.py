"""SQL text binds must be parsed before PostgreSQL casts are applied."""

from pathlib import Path


def test_document_id_binds_use_cast_syntax_that_sqlalchemy_can_parse() -> None:
    source = Path("src/app/features/documents/adapters/persistence/postgres/document_repository.py").read_text(encoding="utf-8")

    assert ":document_id::uuid" not in source
    assert source.count("CAST(:document_id AS uuid)") >= 2
