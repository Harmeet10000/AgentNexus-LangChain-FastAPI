from datetime import datetime
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from app.features.auth.model import UserRole

T = TypeVar("T")


class PaginatedData(BaseModel, Generic[T]):
    """Generic pagination envelope — promoted to shared if used outside users feature."""

    model_config = ConfigDict(frozen=True, populate_by_name=True, serialize_by_alias=True)

    items: list[T]
    total: int
    page: int = Field(ge=1)
    per_page: int = Field(serialization_alias="perPage")
    pages: int


class UserAdminResponse(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True, serialize_by_alias=True)

    id: str
    email: str
    full_name: str | None = Field(default=None, serialization_alias="fullName")
    role: str
    is_verified: bool = Field(serialization_alias="isVerified")
    is_active: bool = Field(serialization_alias="isActive")
    created_at: datetime = Field(serialization_alias="createdAt")
    updated_at: datetime = Field(serialization_alias="updatedAt")
    oauth_providers: list[str] = Field(
        default_factory=list,
        serialization_alias="oauthProviders",
    )


class UpdateUserRoleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: UserRole


class ImpersonateResponse(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True, serialize_by_alias=True)

    access_token: str = Field(serialization_alias="accessToken")
    token_type: str = Field(default="bearer", serialization_alias="tokenType")
    expires_in: int = Field(serialization_alias="expiresIn")
    impersonating_user_id: str = Field(serialization_alias="impersonatingUserId")
