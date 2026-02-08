# app/features/auth/security.py
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from jose import jwt
from passlib.context import CryptContext

from app.config.settings import get_settings

settings = get_settings()

SECRET_KEY = settings.JWT_SECRET_KEY
ALGORITHM = settings.JWT_ALGORITHM

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, hash: str) -> bool:
    return pwd_context.verify(password, hash)


def create_token(
    *,
    user_id: str,
    email: str,
    token_type: str,
    expires_minutes: int,
):
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "email": email,
        "type": token_type,
        "jti": str(uuid4()),
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=expires_minutes)).timestamp()),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

