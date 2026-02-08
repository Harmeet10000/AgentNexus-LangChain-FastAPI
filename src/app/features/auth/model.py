from datetime import datetime, timezone
from typing import Annotated

from beanie import Document, Indexed
from pydantic import EmailStr


class User(Document):
    email: Annotated[EmailStr, Indexed(unique=True)]
    password_hash: str
    full_name: str
    created_at: datetime = datetime.now(timezone.utc)
    updated_at: datetime = datetime.now(timezone.utc)

    class Settings:
        name = "users"
