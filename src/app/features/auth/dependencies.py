# app/features/auth/dependency.py
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from app.config.settings import get_settings
from app.connections.mongodb import get_mongodb
from app.connections.redis import get_redis
from app.features.auth.repository import RefreshTokenRepository, UserRepository
from app.features.auth.service import AuthService

security = HTTPBearer()
settings = get_settings()


def get_user_repository(db=Depends(get_mongodb)) -> UserRepository:
    return UserRepository(db)


def get_refresh_token_repository(redis=Depends(get_redis)) -> RefreshTokenRepository:
    return RefreshTokenRepository(redis)


def get_auth_service(
    user_repo=Depends(get_user_repository),
    refresh_token_repo=Depends(get_refresh_token_repository),
) -> AuthService:
    return AuthService(user_repo, refresh_token_repo)


async def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(security),
    user_repo: UserRepository = Depends(get_user_repository),
):
    try:
        payload = jwt.decode(
            token=creds.credentials,
            key=settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    if payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Invalid token type")

    user = await user_repo.get_by_id(payload["sub"])
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return user
