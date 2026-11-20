"""Commands and queries exchanged by profile use cases."""

from pydantic import BaseModel, ConfigDict

from app.features.profile.domain.entities.profile import Profile


class _ImmutableDTO(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GetProfileQuery(_ImmutableDTO):
    user_id: str


class UpdateProfileCommand(_ImmutableDTO):
    profile: Profile
    full_name: str | None = None


class ChangePasswordCommand(_ImmutableDTO):
    profile: Profile
    current_password: str
    new_password: str
    current_session_id: str | None
    revoke_other_sessions: bool


class UploadAvatarCommand(_ImmutableDTO):
    profile: Profile
    file_data: bytes
    content_type: str


class UploadedAvatar(_ImmutableDTO):
    avatar_url: str
