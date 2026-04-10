from pydantic import BaseModel, ConfigDict, Field


class RequestMeta(BaseModel):
    """Request context echoed back in API responses."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        serialize_by_alias=True,
    )

    ip: str | None = Field(default=None)
    method: str | None = Field(default=None)
    url: str | None = Field(default=None)
    correlation_id: str | None = Field(default=None, serialization_alias="correlationId")


class ErrorDetail(BaseModel):
    """Normalized error payload for non-success responses."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        serialize_by_alias=True,
    )

    code: str
    message: str
    data: dict[str, object] | list[object] | str | None = Field(default=None)
    trace: str | None = Field(default=None)
    inner_error: str | None = Field(default=None, serialization_alias="innerError")
    flow: str | None = Field(default=None)


class APIResponse[T](BaseModel):
    """Default API response envelope for all HTTP handlers."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        serialize_by_alias=True,
    )

    success: bool = Field(default=True)
    status_code: int = Field(default=200, serialization_alias="statusCode")
    request: RequestMeta
    message: str = Field(default="Success")
    data: T | None = Field(default=None)
    error: ErrorDetail | None = Field(default=None)
