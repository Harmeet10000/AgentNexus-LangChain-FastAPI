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

from app.features.auth.adapters.persistence.mongodb.models import (
    UserDocument,
    to_domain_user,
)
from app.features.auth.domain.errors import AuthInfrastructureError, AuthValidationError
from app.observability.logging import logger, trace_layer

if TYPE_CHECKING:
    from app.features.auth.domain.errors import AuthResult
    from app.features.auth.domain.user import User, UserRole


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
        document = await UserDocument.get(PydanticObjectId(user_id))
        return Success(to_domain_user(document) if document is not None else None)
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
        document = await UserDocument.get(PydanticObjectId(user.id))
        if document is None:
            return Failure(
                AuthInfrastructureError(
                    message="User disappeared before role update",
                    details={"user_id": user.id},
                    source="auth_identity",
                )
            )
        await document.update(
            Set(
                {
                    UserDocument.role: role,
                    UserDocument.session_revocation_pending: True,
                    UserDocument.session_revocation_pending_at: datetime.now(UTC),
                    UserDocument.updated_at: datetime.now(UTC),
                }
            ),
            Inc({UserDocument.role_revision: 1}),
        )
        updated = await UserDocument.get(document.id)
        if updated is None:
            return Failure(
                AuthInfrastructureError(
                    message="User disappeared after role update",
                    details={"user_id": str(user.id)},
                    source="auth_identity",
                )
            )
        return Success(to_domain_user(updated))
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
        document_id = PydanticObjectId(user.id)
        if expected_revision is None:
            document = await UserDocument.get(document_id)
            if document is None:
                return Failure(
                    AuthInfrastructureError(
                        message="User disappeared while clearing session revocation state",
                        details={"user_id": user.id},
                        source="auth_identity",
                    )
                )
            await document.update(
                Set(
                    {
                        UserDocument.session_revocation_pending: False,
                        UserDocument.session_revocation_pending_at: None,
                        UserDocument.updated_at: datetime.now(UTC),
                    }
                )
            )
            refreshed = await UserDocument.get(document_id)
            return Success(to_domain_user(refreshed or document))

        updated = await UserDocument.find_one(
            UserDocument.id == document_id,
            UserDocument.role_revision == expected_revision,
            UserDocument.session_revocation_pending == True,  # noqa: E712 — Mongo predicate
        ).update(
            Set(
                {
                    UserDocument.session_revocation_pending: False,
                    UserDocument.session_revocation_pending_at: None,
                    UserDocument.updated_at: datetime.now(UTC),
                }
            ),
            response_type=UpdateResponse.NEW_DOCUMENT,
        )
        if updated is not None:
            return Success(to_domain_user(updated))
        # A newer role change won the race; preserve its pending marker so the
        # newer revision's session invalidation remains retryable.
        current = await UserDocument.get(document_id)
        if current is None:
            return Failure(
                AuthInfrastructureError(
                    message="User disappeared while clearing session revocation state",
                    details={"user_id": str(user.id)},
                    source="auth_identity",
                )
            )
        return Success(to_domain_user(current))
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
        document = await UserDocument.get(PydanticObjectId(user.id))
        if document is None:
            return Failure(
                AuthInfrastructureError(
                    message="User disappeared before active-state update",
                    details={"user_id": user.id},
                    source="auth_identity",
                )
            )
        await document.update(
            Set(
                {
                    UserDocument.is_active: is_active,
                    UserDocument.updated_at: datetime.now(UTC),
                }
            )
        )
        refreshed = await UserDocument.get(document.id)
        return Success(to_domain_user(refreshed or document))
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
        document = await UserDocument.get(PydanticObjectId(user.id))
        if document is not None:
            await document.delete()
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
