from typing import Annotated

from fastapi import Depends

from app.connections import get_redis
from app.features.auth.dependencies import get_refresh_token_repository
from app.features.auth.repository import RefreshTokenRepository
from app.features.users.repository import UserAdminRepository
from app.features.users.service import UserAdminService


async def get_user_admin_repository() -> UserAdminRepository:
    return UserAdminRepository()


async def get_user_admin_service(
    user_repo: Annotated[UserAdminRepository, Depends(get_user_admin_repository)],
    token_repo: Annotated[RefreshTokenRepository, Depends(get_refresh_token_repository)],
) -> UserAdminService:
    return UserAdminService(user_repo, token_repo)


UserAdminServiceDep = Annotated[UserAdminService, Depends(get_user_admin_service)]
