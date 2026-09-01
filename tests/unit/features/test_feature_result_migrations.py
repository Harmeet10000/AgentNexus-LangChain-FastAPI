"""Regression coverage for the section 15 feature Result migrations."""

from __future__ import annotations

from types import SimpleNamespace
from typing import get_args
from unittest.mock import AsyncMock

import pytest
from fastapi import Response
from pydantic import ValidationError
from returns.result import Failure, Success

from app.features.agent_saul.errors import AgentSaulError
from app.features.agent_saul.service import AgentSaulService
from app.features.audit.errors import AuditCode, AuditInfrastructureError
from app.features.dunning.errors import DunningInfrastructureError
from app.features.dunning.service import DunningService
from app.features.ingestion.errors import IngestionGraphError, IngestionPipelineError
from app.features.ingestion.service import IngestionService
from app.features.plans.errors import PlanConflictError
from app.features.plans.service import PlanService
from app.features.profile.errors import ProfileAuthenticationError
from app.features.profile.service import ProfileService
from app.features.users.errors import UsersConflictError, UsersInfrastructureError
from app.features.users.service import UserAdminService
from app.shared.result import render_result


def test_error_classification_is_not_constructible_or_serialized() -> None:
    error = AuditInfrastructureError(
        message="audit failed",
        source="test",
        operation="create",
    )

    assert error.code is AuditCode.DATABASE_ERROR
    assert error.model_dump() == {
        "message": "audit failed",
        "details": None,
        "source": "test",
        "operation": "create",
    }
    with pytest.raises(ValidationError):
        AuditInfrastructureError.model_validate(
            {"message": "audit failed", "operation": "create", "code": "TYPO"}
        )


def test_all_requested_features_expose_closed_error_unions() -> None:
    members = get_args(AgentSaulError.__value__)
    assert members
    assert all(error_type.__base__.__name__ == "FeatureError" for error_type in members)


async def test_users_self_role_change_is_a_renderable_conflict() -> None:
    service = UserAdminService(user_repo=AsyncMock(), token_repo=AsyncMock())
    result = await service.update_role("same", SimpleNamespace(), "same")
    assert isinstance(result, Failure)
    assert isinstance(result.failure(), UsersConflictError)

    response = Response()
    envelope = render_result(result, response, message="User role updated")
    assert response.status_code == 409
    assert envelope.success is False


async def test_users_translate_auth_session_failure_locally() -> None:
    user = SimpleNamespace(id="target", is_active=True)
    user_repo = SimpleNamespace(
        find_by_id=AsyncMock(return_value=Success(user)),
        set_active=AsyncMock(return_value=Success(user)),
    )
    auth_error = SimpleNamespace(message="redis unavailable", details={"backend": "redis"})
    token_repo = SimpleNamespace(
        revoke_all_user_sessions=AsyncMock(return_value=Failure(auth_error))
    )
    service = UserAdminService(user_repo=user_repo, token_repo=token_repo)

    result = await service.set_active("target", is_active=False, requesting_admin_id="admin")
    assert isinstance(result, Failure)
    assert isinstance(result.failure(), UsersInfrastructureError)


async def test_ingestion_graph_exception_returns_typed_failure() -> None:
    graph = SimpleNamespace(ainvoke=AsyncMock(side_effect=RuntimeError("graph down")))
    result = await IngestionService(graph).ingest_document(b"x", "user", "a.pdf", "a.pdf")
    assert isinstance(result, Failure)
    assert isinstance(result.failure(), IngestionGraphError)


async def test_ingestion_state_failure_is_translated_locally() -> None:
    graph = SimpleNamespace(ainvoke=AsyncMock(return_value={"failure": {"message": "node failed"}}))
    result = await IngestionService(graph).ingest_document(b"x", "user", "a.pdf", "a.pdf")
    assert isinstance(result, Failure)
    assert isinstance(result.failure(), IngestionPipelineError)


async def test_dunning_propagates_subscription_update_failure() -> None:
    subscription = SimpleNamespace(
        id="sub-1",
        metadata_={},
        retry_count=0,
        max_retries=3,
        version=1,
    )
    collaborator_error = SimpleNamespace(message="update failed", details={"id": "sub-1"})
    subscriptions = SimpleNamespace(
        update_with_lock=AsyncMock(return_value=Failure(collaborator_error))
    )
    service = DunningService(
        session=AsyncMock(),
        subscriptions=subscriptions,
        plans=AsyncMock(),
        audit=AsyncMock(),
    )
    service._attempt_charge = AsyncMock(return_value=Success({"status": "skipped"}))
    service._next_retry_at = lambda _subscription, *, now: now

    result = await service.execute_retry(subscription)
    assert isinstance(result, Failure)
    assert isinstance(result.failure(), DunningInfrastructureError)


async def test_plan_duplicate_is_returned_without_calling_create() -> None:
    plans = SimpleNamespace(
        find_by_name=AsyncMock(return_value=Success(SimpleNamespace())),
        create=AsyncMock(),
    )
    service = PlanService(AsyncMock(), plans, AsyncMock())
    dto = SimpleNamespace(name="Pro")
    result = await service.create_plan(dto, user_id="admin")
    assert isinstance(result, Failure)
    assert isinstance(result.failure(), PlanConflictError)
    plans.create.assert_not_awaited()


async def test_profile_invalid_current_password_is_typed() -> None:
    service = ProfileService(AsyncMock(), AsyncMock(), None)
    user = SimpleNamespace(hashed_password="not-a-valid-hash")
    result = await service.change_password(user, "wrong", "new", None, revoke_other_sessions=False)
    assert isinstance(result, Failure)
    assert isinstance(result.failure(), ProfileAuthenticationError)


def test_agent_saul_http_preflight_returns_result() -> None:
    result = AgentSaulService.create_session(
        websocket_url="wss://example.test/ws/thread-1",
        thread_id="thread-1",
    )
    assert isinstance(result, Success)
    assert result.unwrap().thread_id == "thread-1"
