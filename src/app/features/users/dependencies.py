from typing import Annotated

from fastapi import Depends

from app.features.auth import RefreshTokenRepository, get_refresh_token_repository

from .repository import UserAdminRepository
from .service import UserAdminService


async def get_user_admin_repository() -> UserAdminRepository:
    return UserAdminRepository()


async def get_user_admin_service(
    user_repo: Annotated[UserAdminRepository, Depends(get_user_admin_repository)],
    token_repo: Annotated[RefreshTokenRepository, Depends(get_refresh_token_repository)],
) -> UserAdminService:
    return UserAdminService(user_repo, token_repo)


UserAdminServiceDep = Annotated[UserAdminService, Depends(get_user_admin_service)]
