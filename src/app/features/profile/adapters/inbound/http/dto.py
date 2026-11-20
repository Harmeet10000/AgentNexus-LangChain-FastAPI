"""HTTP request and response DTOs for the profile resource."""

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator


class UpdateProfileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    full_name: Annotated[str | None, Field(default=None, min_length=1, max_length=256)]


class ChangePasswordRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    current_password: Annotated[str, Field(min_length=1, max_length=128)]
    new_password: Annotated[str, Field(min_length=8, max_length=128)]
    revoke_other_sessions: bool = Field(default=True)

    @field_validator("new_password")
    @classmethod
    def password_strong_enough(cls, value: str) -> str:
        if not any(character.isupper() for character in value):
            message = "Must contain at least one uppercase letter"
            raise ValueError(message)
        if not any(character.isdigit() for character in value):
            message = "Must contain at least one digit"
            raise ValueError(message)
        return value


class AvatarResponse(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True, serialize_by_alias=True)

    avatar_url: str = Field(serialization_alias="avatarUrl")


class ProfileResponse(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True, serialize_by_alias=True)

    id: str
    email: str
    full_name: str | None = Field(default=None, serialization_alias="fullName")
    avatar_url: str | None = Field(default=None, serialization_alias="avatarUrl")
    role: str
    is_verified: bool = Field(serialization_alias="isVerified")
    is_active: bool = Field(serialization_alias="isActive")
    created_at: datetime = Field(serialization_alias="createdAt")
