"""Contract tests for the documents and auth feature error migrations."""

from pathlib import Path

import pytest
from fastapi import Response
from pydantic import ValidationError
from pymongo.errors import PyMongoError
from returns.result import Failure

from app.features.auth.dependencies import raise_auth_error
from app.features.auth.errors import (
    AuthAuthenticationError,
    AuthAuthorizationError,
    AuthInfrastructureError,
)
from app.features.auth.repository import UserRepository
from app.features.documents.errors import DocumentDatabaseError, DocumentNotFoundError
from app.shared.rag.document_processing.embedder import _provider_failure
from app.shared.rag.errors import RagProviderError
from app.shared.result import ErrorKind, render_result
from app.utils import ForbiddenException, UnauthorizedException

_AUTH_REPOSITORY = Path("src/app/features/auth/repository.py")


def test_feature_classification_is_constant_and_not_serialized() -> None:
    error = AuthAuthenticationError(message="Invalid credentials")

    assert error.kind is ErrorKind.AUTHENTICATION
    assert error.model_dump() == {"message": "Invalid credentials", "details": None, "source": None}
    with pytest.raises(ValidationError):
        AuthAuthenticationError(message="Invalid credentials", code="WRONG")


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (AuthAuthenticationError(message="bad token"), 401),
        (AuthAuthorizationError(message="denied"), 403),
        (AuthInfrastructureError(message="redis unavailable"), 503),
        (DocumentNotFoundError(message="missing"), 404),
        (DocumentDatabaseError(message="database failed"), 500),
    ],
)
def test_render_result_preserves_security_and_store_statuses(
    error: object, expected_status: int
) -> None:
    response = Response()

    render_result(Failure(error), response, message="unused")

    assert response.status_code == expected_status


def test_auth_dependency_translates_authentication_and_authorization() -> None:
    with pytest.raises(UnauthorizedException) as authentication:
        raise_auth_error(AuthAuthenticationError(message="bad token"))
    with pytest.raises(ForbiddenException) as authorization:
        raise_auth_error(AuthAuthorizationError(message="denied"))

    assert authentication.value.status_code == 401
    assert authorization.value.status_code == 403


async def test_mongo_failure_is_retryable_without_rollback(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fail_find_one(*_args: object, **_kwargs: object) -> object:
        message = "mongo unavailable"
        raise PyMongoError(message)

    monkeypatch.setattr("app.features.auth.repository.User.email", "email", raising=False)
    monkeypatch.setattr("app.features.auth.repository.User.find_one", fail_find_one)

    result = await UserRepository.find_by_email("user@example.com")

    assert isinstance(result, Failure)
    assert isinstance(result.failure(), AuthInfrastructureError)
    assert result.failure().retryable is True


def test_auth_document_store_has_no_rollback() -> None:
    assert "rollback" not in _AUTH_REPOSITORY.read_text(encoding="utf-8")


def test_rag_provider_boundary_returns_its_own_typed_failure() -> None:
    result = _provider_failure("provider unavailable", model="gemini", text_count=2)

    assert isinstance(result, Failure)
    assert isinstance(result.failure(), RagProviderError)
    assert result.failure().model == "gemini"
    assert result.failure().text_count == 2
