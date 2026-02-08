# app/features/auth/dependency.py
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from app.connections.mongodb import get_db
from app.connections.redis import get_redis
from app.features.auth.repository import RefreshTokenRepository, UserRepository
from app.features.auth.security import ALGORITHM, SECRET_KEY
from app.features.auth.service import AuthService

security = HTTPBearer()


def get_user_repository(db=Depends(get_db)) -> UserRepository:
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
            creds.credentials,
            SECRET_KEY,
            algorithms=[ALGORITHM],
        )
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    if payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Invalid token type")

    user = await user_repo.get_by_id(payload["sub"])
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return user
