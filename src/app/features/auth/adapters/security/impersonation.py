"""Security adapter that issues the auth impersonation token."""

from app.security.tokens import create_impersonation_token


class ImpersonationTokenIssuer:
    @staticmethod
    def issue(
        *,
        target_account_id: str,
        target_role: str,
        target_permissions: frozenset[str],
        admin_account_id: str,
    ) -> tuple[str, int]:
        return create_impersonation_token(
            target_user_id=target_account_id,
            target_role=target_role,
            target_permissions=target_permissions,
            admin_user_id=admin_account_id,
        )


__all__ = ["ImpersonationTokenIssuer"]
