from typing import Generic, TypeVar

from pydantic import BaseModel, Field

# T represents the specific data payload for any given route
T = TypeVar("T")


class RequestMeta(BaseModel):
    ip: str | None = Field(default=None)
    method: str | None = Field(default=None)
    url: str | None = Field(default=None)
    correlationId: str | None = Field(default=None)


class APIResponse(BaseModel, Generic[T]):
    success: bool = Field(default=True)
    statusCode: int = Field(default=200)
    request: RequestMeta
    message: str = Field(default="Success")
    data: T | None = Field(default=None)

# usage example:
# class UserSchema(BaseModel):
#     id: int
#     name: str

# # Notice we define the response_model using the Generic envelope
# @router.get("/users/{user_id}", response_model=APIResponse[UserSchema])
# async def get_user(user_id: int, request: Request):
