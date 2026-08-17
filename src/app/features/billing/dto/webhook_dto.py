"""Webhook request/response DTOs."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class WebhookPayload(BaseModel):
    """Raw Razorpay webhook payload."""

    model_config = ConfigDict(extra="allow")

    entity: str = Field(default="event")
    account_id: str | None = Field(default=None, serialization_alias="accountId")
    event: str
    contains: list[str] = Field(default_factory=list)
    payload: dict[str, object] = Field(default_factory=dict)
    created_at: int = Field(default=0, serialization_alias="createdAt")


class WebhookEventDTO(BaseModel):
    """Normalized webhook event passed to the webhook service."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    event_type: str
    payload: dict[str, object]
    signature: str
