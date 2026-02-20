# app/features/auth/security.py
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from jose import jwt
from passlib.context import CryptContext

from app.config.settings import get_settings

settings = get_settings()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(secret=password)


def verify_password(password: str, hash: str) -> bool:
    return pwd_context.verify(secret=password, hash=hash)


def create_token(
    *,
    user_id: str,
    email: str,
    token_type: str,
    expires_minutes: int,
):
    now = datetime.now(UTC)
    payload = {
        "sub": user_id,
        "email": email,
        "type": token_type,
        "jti": str(uuid4()),
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=expires_minutes)).timestamp()),
    }
    return jwt.encode(claims=payload, key=settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
