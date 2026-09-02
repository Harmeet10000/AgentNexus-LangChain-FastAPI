"""Runtime coverage for SQLAlchemy exception notes."""

from types import SimpleNamespace

from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.shared.result.diagnostics import add_database_error_note


def test_integrity_error_note_includes_constraint_and_context() -> None:
    original = SimpleNamespace(diag=SimpleNamespace(constraint_name="uq_payments_external_id"))
    error = IntegrityError("insert", {}, original)

    add_database_error_note(
        error,
        table="payments",
        operation="create",
        context={"payment_id": "p-1"},
    )

    assert error.__notes__ == [
        "table=payments, operation=create, query=create, "
        "constraint_name=uq_payments_external_id, payment_id=p-1"
    ]


def test_generic_database_error_note_is_bounded() -> None:
    error = SQLAlchemyError("database unavailable")

    add_database_error_note(
        error,
        table="chunks, documents",
        operation="vector_search",
        context={"embedding_dim": 1536, "query": "x" * 200},
    )

    assert "table=chunks, documents" in error.__notes__[0]
    assert "operation=vector_search" in error.__notes__[0]
    assert "query=" in error.__notes__[0]
    assert len(error.__notes__[0]) < 400
