from pydantic import BaseModel, ConfigDict, EmailStr, Field

# Shared configuration for response models
# 'slots' reduces memory footprint, 'from_attributes' allows ORM compatibility
response_config = ConfigDict(from_attributes=True, frozen=True)


class RegisterRequest(BaseModel):
    # email validation is already robust, but we can add strictness
    email: EmailStr
    # password usually needs a min_length for security;
    # setting it here prevents useless DB calls for empty strings
    password: str = Field(min_length=8, max_length=100)
    full_name: str = Field(min_length=1, max_length=100)


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")  # Security: don't allow extra noise
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    model_config = response_config
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    model_config = response_config
    id: str
    email: EmailStr
    full_name: str


class LogoutResponse(BaseModel):
    # For very simple responses, we can skip complex validation
    model_config = ConfigDict()
    detail: str
