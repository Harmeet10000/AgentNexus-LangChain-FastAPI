"""Password hashing and one-way token digests. No domain imports."""

import hashlib
import secrets

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

_ph = PasswordHasher(
    time_cost=2,
    memory_cost=65536,  # 64 MB
    parallelism=2,
    hash_len=32,
    salt_len=16,
)


def hash_password(password: str) -> str:
    return _ph.hash(password)


def verify_password(hashed: str, plain: str) -> bool:
    try:
        return _ph.verify(hashed, plain)
    except VerifyMismatchError:
        return False
    except (VerificationError, InvalidHashError):
        return False


def needs_rehash(hashed: str) -> bool:
    """True when stored hash was created with weaker argon2 params than current config."""
    return _ph.check_needs_rehash(hashed)


def generate_token() -> str:
    """Cryptographically secure URL-safe token for email delivery."""
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    """SHA-256 digest stored in DB; the raw token travels only via email."""
    return hashlib.sha256(token.encode()).hexdigest()
