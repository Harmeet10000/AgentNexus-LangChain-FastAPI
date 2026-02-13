from datetime import UTC, datetime
from typing import Annotated

from beanie import Document, Indexed
from pydantic import EmailStr


class User(Document):
    email: Annotated[EmailStr, Indexed(unique=True)]
    password_hash: str
    full_name: str
    created_at: datetime = datetime.now(UTC)
    updated_at: datetime = datetime.now(UTC)

    class Settings:
        name = "users"
        indexes = [
            [("created_at", -1)],
        ]
