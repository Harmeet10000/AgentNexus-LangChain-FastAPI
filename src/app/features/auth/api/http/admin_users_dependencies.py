"""FastAPI composition for auth administrative account operations."""

from typing import Annotated

from fastapi import Depends

from app.features.auth.adapters.persistence.mongodb.account_repository import RefreshTokenRepository
from app.features.auth.adapters.persistence.mongodb.admin_accounts import AdminAccountRepository
from app.features.auth.adapters.security.impersonation import ImpersonationTokenIssuer
from app.features.auth.api.http.dependencies import (
    get_refresh_token_repository,
)
from app.features.auth.application.commands.admin_accounts import AdminAccounts
from app.features.auth.application.queries.admin_users import AdminUsersQuery


async def get_admin_account_repository() -> AdminAccountRepository:
    return AdminAccountRepository()


async def get_admin_users_query(
    repository: Annotated[AdminAccountRepository, Depends(get_admin_account_repository)],
) -> AdminUsersQuery:
    return AdminUsersQuery(repository)


async def get_admin_accounts(
    repository: Annotated[AdminAccountRepository, Depends(get_admin_account_repository)],
    token_repository: Annotated[RefreshTokenRepository, Depends(get_refresh_token_repository)],
) -> AdminAccounts:
    return AdminAccounts(
        repository,
        token_repository,
        issuer=ImpersonationTokenIssuer(),
    )


AdminUsersQueryDep = Annotated[AdminUsersQuery, Depends(get_admin_users_query)]
AdminAccountsDep = Annotated[AdminAccounts, Depends(get_admin_accounts)]


__all__ = ["AdminAccountsDep", "AdminUsersQueryDep"]
