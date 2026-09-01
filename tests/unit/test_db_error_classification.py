"""Relational and document-store errors retain opposite retry semantics."""

from app.features.auth.errors import AuthInfrastructureError
from app.features.documents.errors import DocumentDatabaseError
from app.features.plans.errors import PlanInfrastructureError


def test_relational_database_errors_are_not_retryable() -> None:
    errors = (
        DocumentDatabaseError(message="document database failed"),
        PlanInfrastructureError(message="plan database failed", operation="find"),
    )

    assert all(error.retryable is False for error in errors)


def test_auth_document_store_errors_remain_retryable() -> None:
    error = AuthInfrastructureError(message="MongoDB unavailable")

    assert error.retryable is True
