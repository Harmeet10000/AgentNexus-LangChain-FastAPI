"""Runtime coverage for SQLAlchemy exception notes."""

from types import SimpleNamespace

from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.kernel.result import add_database_error_note


def test_integrity_error_note_includes_constraint_context_and_traceback() -> None:
    original = SimpleNamespace(diag=SimpleNamespace(constraint_name="uq_payments_external_id"))
    error = IntegrityError("insert", {}, original)

    try:
        raise error
    except SQLAlchemyError:
        add_database_error_note(
            error,
            table="payments",
            operation="create",
            context={"payment_id": "p-1"},
        )

    assert error.__notes__[:1] == [
        "table=payments, operation=create, query=create, "
        "constraint_name=uq_payments_external_id, payment_id=p-1"
    ]
    # The formatted traceback rides the second note: the typed Failure that
    # replaces this exception discards the original, so this is the only
    # durable record of where the driver error happened.
    assert "Traceback" in error.__notes__[1]


def test_note_outside_an_except_block_adds_no_traceback() -> None:
    """traceback.format_exc() is meaningless without an active exception."""
    error = IntegrityError("insert", {}, SimpleNamespace())

    add_database_error_note(error, table="payments", operation="create")

    assert len(error.__notes__) == 1


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
