from datetime import datetime, timezone

from fastapi import HTTPException
from jose import JWTError, jwt

from app.config.settings import get_settings
from app.features.auth.dto import RegisterRequest
from app.features.auth.model import User
from app.features.auth.repository import RefreshTokenRepository, UserRepository
from app.features.auth.security import (
    ALGORITHM,
    SECRET_KEY,
    create_token,
    hash_password,
    verify_password,
)
from app.utils.logger import logger

settings = get_settings()


class AuthService:
    def __init__(self, user_repo: UserRepository, refresh_token_repo: RefreshTokenRepository):
        self.user_repo = user_repo
        self.refresh_token_repo = refresh_token_repo

    async def register(self, data: RegisterRequest):
        try:
            if await self.user_repo.get_by_email(data.email):
                raise ValueError("Email already exists")

            user = User(
                email=data.email,
                password_hash=hash_password(data.password),
                full_name=data.full_name,
            )
            await self.user_repo.create(user)
            logger.info(f"User registered successfully: {data.email}")
            return user
        except ValueError as e:
            logger.warning(f"Registration failed for {data.email}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during registration for {data.email}: {str(e)}", exc_info=True)
            raise

    async def login(self, email: str, password: str):
        try:
            user = await self.user_repo.get_by_email(email)
            if not user or not verify_password(password, user.password_hash):
                logger.warning(f"Failed login attempt for email: {email}")
                raise HTTPException(status_code=401, detail="Invalid credentials")

            access = create_token(
                user_id=str(user.id),
                email=user.email,
                token_type="access",
                expires_minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
            )

            refresh = create_token(
                user_id=str(user.id),
                email=user.email,
                token_type="refresh",
                expires_minutes=settings.JWT_REFRESH_TOKEN_EXPIRE_MINUTES,
            )

            payload = jwt.decode(refresh, SECRET_KEY, algorithms=[ALGORITHM])
            ttl = payload["exp"] - int(datetime.now(tz=timezone.utc).timestamp())

            await self.refresh_token_repo.store(payload["jti"], payload["sub"], ttl)

            logger.info(f"User logged in successfully: {email}")
            return {
                "access_token": access,
                "refresh_token": refresh,
                "user": user,
                "token_type": "bearer",
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error during login for {email}: {str(e)}", exc_info=True)
            raise

    async def refresh(self, refresh_token: str):
        try:
            logger.info("Attempting to refresh token")
            payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        except JWTError as e:
            logger.warning(f"Invalid refresh token - JWT decode failed: {str(e)}")
            raise HTTPException(status_code=401, detail="Invalid refresh token")
        except Exception as e:
            logger.error(f"Unexpected error decoding refresh token: {str(e)}", exc_info=True)
            raise HTTPException(status_code=401, detail="Invalid refresh token")

        try:
            if payload["type"] != "refresh":
                logger.warning(f"Invalid token type: {payload.get('type')}")
                raise HTTPException(status_code=401, detail="Invalid token type")

            if not await self.refresh_token_repo.exists(payload["jti"]):
                logger.warning(f"Refresh token revoked or not found: {payload['jti']}")
                raise HTTPException(status_code=401, detail="Refresh token revoked")

            # 🔁 ROTATION: revoke old refresh token
            await self.refresh_token_repo.revoke(payload["jti"])

            user = await self.user_repo.get_by_id(payload["sub"])
            if not user:
                logger.warning(f"User not found for token refresh: {payload['sub']}")
                raise HTTPException(status_code=401, detail="User not found")

            # issue new tokens
            logger.info(f"Token refreshed successfully for user: {user.email}")
            return await self.login(user.email, user.password_hash)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error during token refresh: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Token refresh failed")

    async def logout(self, refresh_token: str):
        try:
            payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
            if payload.get("jti"):
                await self.refresh_token_repo.revoke(payload["jti"])
                logger.info(f"User logged out successfully: {payload.get('sub')}")
        except JWTError as e:
            logger.warning(f"Logout with invalid token (idempotent): {str(e)}")
            return  # idempotent logout
        except Exception as e:
            logger.error(f"Unexpected error during logout: {str(e)}", exc_info=True)
            return  # idempotent logout
