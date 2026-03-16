from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

# ── Requests ──────────────────────────────────────────────────────────────────


class RegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: EmailStr
    password: Annotated[str, Field(min_length=8, max_length=128)]
    full_name: Annotated[str | None, Field(default=None, max_length=256)]

    @field_validator("password")
    @classmethod
    def password_strong_enough(cls, v: str) -> str:
        if not any(c.isupper() for c in v):
            raise ValueError("Must contain at least one uppercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Must contain at least one digit")
        return v


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: EmailStr
    password: Annotated[str, Field(min_length=1, max_length=128)]
    device_name: Annotated[str | None, Field(default=None, max_length=128)]


class RefreshRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    refresh_token: str


class LogoutRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    refresh_token: str


class VerifyEmailRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    token: str


class ResendVerificationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: EmailStr


class ForgotPasswordRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: EmailStr


class ResetPasswordRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    token: str
    new_password: Annotated[str, Field(min_length=8, max_length=128)]

    @field_validator("new_password")
    @classmethod
    def password_strong_enough(cls, v: str) -> str:
        if not any(c.isupper() for c in v):
            raise ValueError("Must contain at least one uppercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Must contain at least one digit")
        return v


# ── Responses ─────────────────────────────────────────────────────────────────


class TokenResponse(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True, serialize_by_alias=True)

    access_token: str = Field(serialization_alias="accessToken")
    refresh_token: str = Field(serialization_alias="refreshToken")
    token_type: str = Field(default="bearer", serialization_alias="tokenType")
    expires_in: int = Field(serialization_alias="expiresIn")


class UserResponse(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True, serialize_by_alias=True)

    id: str
    email: str
    full_name: str | None = Field(default=None, serialization_alias="fullName")
    role: str
    is_verified: bool = Field(serialization_alias="isVerified")
    is_active: bool = Field(serialization_alias="isActive")
    created_at: datetime = Field(serialization_alias="createdAt")


class SessionResponse(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True, serialize_by_alias=True)

    session_id: str = Field(serialization_alias="sessionId")
    device_id: str = Field(serialization_alias="deviceId")
    device_name: str | None = Field(default=None, serialization_alias="deviceName")
    ip_address: str | None = Field(default=None, serialization_alias="ipAddress")
    created_at: datetime = Field(serialization_alias="createdAt")
    expires_at: datetime = Field(serialization_alias="expiresAt")
    is_current: bool = Field(default=False, serialization_alias="isCurrent")


class OAuthAuthorizeResponse(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True, serialize_by_alias=True)

    authorization_url: str = Field(serialization_alias="authorizationUrl")
    provider: str
