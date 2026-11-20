"""The Mongo profile projection's persistence shape.

MongoDB account documents remain auth-owned during this phase.  This protocol
names only the fields the profile adapter consumes, avoiding a second Beanie
document or a second source of truth.
"""

from datetime import datetime
from typing import Protocol


class ProfileModel(Protocol):
    id: str
    email: str
    full_name: str | None
    password_hash: str | None
    is_active: bool
    is_verified: bool
    role: object
    avatar_url: str | None
    created_at: datetime
