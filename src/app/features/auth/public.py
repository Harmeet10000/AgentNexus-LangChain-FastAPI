"""Synchronous, transport-neutral surface of the auth feature.

Only immutable auth DTOs and auth application operations are exported here.
The collaborators required by those operations are intentionally not
re-exported; composition belongs to the caller's wiring layer.
"""

from app.features.auth.application.dto.public import (
    AccountRead,
    ImpersonationToken,
    PasswordChange,
    ProfileMutation,
    RoleAdministration,
    SessionRevocation,
    StatusAdministration,
)
from app.features.auth.application.public_operations import (
    change_password,
    get_account_by_email,
    get_account_by_id,
    issue_impersonation_token,
    mutate_profile,
    revoke_all_sessions,
    revoke_session,
    set_account_status,
    update_account_role,
)

__all__ = [
    "AccountRead",
    "ImpersonationToken",
    "PasswordChange",
    "ProfileMutation",
    "RoleAdministration",
    "SessionRevocation",
    "StatusAdministration",
    "change_password",
    "get_account_by_email",
    "get_account_by_id",
    "issue_impersonation_token",
    "mutate_profile",
    "revoke_all_sessions",
    "revoke_session",
    "set_account_status",
    "update_account_role",
]
