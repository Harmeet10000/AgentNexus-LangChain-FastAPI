"""The profile aggregate's framework-independent entity."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class Profile(BaseModel):
    """The account fields the profile slice is allowed to read and mutate.

    Authentication remains the source of truth for account state.  This
    entity is deliberately a projection of that state rather than an auth
    document or a persistence model.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    email: str
    full_name: str | None
    is_active: bool
    is_verified: bool
    role: str
    avatar_url: str | None
    created_at: datetime
