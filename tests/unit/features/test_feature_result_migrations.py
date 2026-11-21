"""Regression coverage for the section 15 feature Result migrations."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import get_args
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError
from returns.result import Failure, Success

from app.features.agent_saul.application.service import AgentSaulService
from app.features.agent_saul.domain.errors import AgentSaulError
from app.features.audit.domain.errors import AuditCode, AuditInfrastructureError
from app.features.auth.domain.user import UserRole
from app.features.billing.dunning.exceptions import DunningInfrastructureError
from app.features.billing.dunning.service import DunningService
from app.features.billing.plans.exceptions import PlanConflictError
from app.features.billing.plans.service import PlanService
from app.features.profile.application.commands.update_profile import (
    UpdateProfileCommandHandler,
)
from app.features.profile.application.dto.profile import ChangePasswordCommand
from app.features.profile.domain.entities.profile import Profile
from app.features.profile.domain.errors import ProfileAuthenticationError


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
    service = UpdateProfileCommandHandler(
        AsyncMock(),
        credentials=SimpleNamespace(
            change_password=AsyncMock(
                return_value=Failure(
                    ProfileAuthenticationError(message="Current password is incorrect")
                )
            )
        ),
        storage=None,
        logger=SimpleNamespace(),
    )
    user = Profile(
        id="507f1f77bcf86cd799439011",
        email="user@example.com",
        full_name=None,
        is_active=True,
        is_verified=True,
        role=UserRole.USER,
        avatar_url=None,
        created_at=datetime.now(UTC),
    )
    result = await service.change_password(
        ChangePasswordCommand(
            profile=user,
            current_password="wrong",
            new_password="new",
            current_session_id=None,
            revoke_other_sessions=False,
        )
    )
    assert isinstance(result, Failure)
    assert isinstance(result.failure(), ProfileAuthenticationError)


def test_agent_saul_http_preflight_returns_result() -> None:
    result = AgentSaulService.create_session(
        websocket_url="wss://example.test/ws/thread-1",
        thread_id="thread-1",
    )
    assert isinstance(result, Success)
    assert result.unwrap().thread_id == "thread-1"
