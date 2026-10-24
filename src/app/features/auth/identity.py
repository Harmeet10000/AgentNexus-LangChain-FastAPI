"""Identity persistence ownership.

The `User` document is owned by the auth feature: credential lifecycle and
all raw document reads/writes go through this module. The users feature keeps
administrative authorization and use cases (role, activation, deletion,
listing) and translates identity failures into its own error union — it never
touches the driver directly for plain document operations.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from beanie import PydanticObjectId
from beanie.odm.queries.update import UpdateResponse
from beanie.operators import Inc, Set
from pymongo.errors import PyMongoError
from returns.result import Failure, Success

from app.observability.logging import logger, trace_layer

from .exceptions import AuthInfrastructureError, AuthValidationError
from .models import User

if TYPE_CHECKING:
    from .exceptions import AuthResult
    from .models import UserRole


@trace_layer("repository")
async def get_user_document(user_id: str) -> AuthResult[User | None]:
    """Read one user document by id. None when absent — absence is not a failure."""
    if not PydanticObjectId.is_valid(user_id):
        return Failure(
            AuthValidationError(
                message="Invalid user identifier",
                details={"user_id": user_id},
                source="auth_identity",
            )
        )
    try:
        return Success(await User.get(PydanticObjectId(user_id)))
    except PyMongoError as exc:
        exc.add_note(f"user_id={user_id}, operation=get_user_document")
        logger.bind(operation="get_user_document", user_id=user_id, error=str(exc)).exception(
            "auth_identity_failed"
        )
        return Failure(
            AuthInfrastructureError(
                message="User lookup failed",
                details={"user_id": user_id, "error": str(exc)},
                source="auth_identity",
            )
        )


@trace_layer("repository")
async def set_user_role(user: User, role: UserRole) -> AuthResult[User]:
    """Apply an administrative role change plus `updated_at`, preserving identity."""
    try:
        await user.update(
            Set(
                {
                    User.role: role,
                    User.session_revocation_pending: True,
                    User.session_revocation_pending_at: datetime.now(UTC),
                    User.updated_at: datetime.now(UTC),
                }
            ),
            Inc({User.role_revision: 1}),
        )
        updated = await User.get(user.id)
        if updated is None:
            return Failure(
                AuthInfrastructureError(
                    message="User disappeared after role update",
                    details={"user_id": str(user.id)},
                    source="auth_identity",
                )
            )
        return Success(updated)
    except PyMongoError as exc:
        exc.add_note(f"user_id={user.id}, operation=set_user_role")
        logger.bind(operation="set_user_role", user_id=str(user.id), error=str(exc)).exception(
            "auth_identity_failed"
        )
        return Failure(
            AuthInfrastructureError(
                message="User role update failed",
                details={"user_id": str(user.id), "error": str(exc)},
                source="auth_identity",
            )
        )


@trace_layer("repository")
async def clear_session_revocation_pending(
    user: User, *, expected_revision: int | None = None
) -> AuthResult[User]:
    """Mark role-change session invalidation complete after Redis commits."""
    try:
        if expected_revision is None:
            await user.update(
                Set(
                    {
                        User.session_revocation_pending: False,
                        User.session_revocation_pending_at: None,
                        User.updated_at: datetime.now(UTC),
                    }
                )
            )
            user.session_revocation_pending = False
            return Success(user)

        updated = await User.find_one(
            User.id == user.id,
            User.role_revision == expected_revision,
            User.session_revocation_pending == True,  # noqa: E712 — Mongo predicate
        ).update(
            Set(
                {
                    User.session_revocation_pending: False,
                    User.session_revocation_pending_at: None,
                    User.updated_at: datetime.now(UTC),
                }
            ),
            response_type=UpdateResponse.NEW_DOCUMENT,
        )
        if updated is not None:
            return Success(updated)
        # A newer role change won the race; preserve its pending marker so the
        # newer revision's session invalidation remains retryable.
        current = await User.get(user.id)
        if current is None:
            return Failure(
                AuthInfrastructureError(
                    message="User disappeared while clearing session revocation state",
                    details={"user_id": str(user.id)},
                    source="auth_identity",
                )
            )
        return Success(current)
    except PyMongoError as exc:
        exc.add_note(f"user_id={user.id}, operation=clear_session_revocation_pending")
        logger.bind(
            operation="clear_session_revocation_pending", user_id=str(user.id), error=str(exc)
        ).exception("auth_identity_failed")
        return Failure(
            AuthInfrastructureError(
                message="Session revocation state update failed",
                details={"user_id": str(user.id), "error": str(exc)},
                source="auth_identity",
            )
        )


@trace_layer("repository")
async def set_user_active(user: User, *, is_active: bool) -> AuthResult[User]:
    """Apply an administrative activation change plus `updated_at`."""
    try:
        await user.update(Set({User.is_active: is_active, User.updated_at: datetime.now(UTC)}))
        user.is_active = is_active
        return Success(user)
    except PyMongoError as exc:
        exc.add_note(f"user_id={user.id}, operation=set_user_active")
        logger.bind(operation="set_user_active", user_id=str(user.id), error=str(exc)).exception(
            "auth_identity_failed"
        )
        return Failure(
            AuthInfrastructureError(
                message="User active state update failed",
                details={"user_id": str(user.id), "error": str(exc)},
                source="auth_identity",
            )
        )


@trace_layer("repository")
async def delete_user_document(user: User) -> AuthResult[None]:
    """Hard-delete one user document."""
    try:
        await user.delete()
        return Success(None)
    except PyMongoError as exc:
        exc.add_note(f"user_id={user.id}, operation=delete_user_document")
        logger.bind(
            operation="delete_user_document", user_id=str(user.id), error=str(exc)
        ).exception("auth_identity_failed")
        return Failure(
            AuthInfrastructureError(
                message="User deletion failed",
                details={"user_id": str(user.id), "error": str(exc)},
                source="auth_identity",
            )
        )
