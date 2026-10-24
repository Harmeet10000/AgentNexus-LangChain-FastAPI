"""OAuth provider account policy. Stays in the auth feature.

Provider configuration and userinfo normalization are account policy, not
reusable cryptography — the stateless state signing they rely on lives in
`app.security.tokens`.
"""

from typing import TYPE_CHECKING

from authlib.integrations.httpx_client import AsyncOAuth2Client
from pydantic import BaseModel, ConfigDict

from app.config import get_settings
from app.utils import ValidationException

if TYPE_CHECKING:
    from typing import Any

_SUPPORTED_PROVIDERS = frozenset({"google", "github"})


class OAuthProviderConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    client_id: str
    client_secret: str
    authorization_endpoint: str
    token_endpoint: str
    scope: str
    redirect_uri: str


def get_oauth_config(provider: str) -> OAuthProviderConfig:
    if provider not in _SUPPORTED_PROVIDERS:
        msg = f"Unsupported OAuth provider: {provider}"
        raise ValidationException(msg)

    settings = get_settings()
    base = settings.BACKEND_URL.rstrip("/")

    match provider:
        case "google":
            return OAuthProviderConfig(
                client_id=settings.GOOGLE_CLIENT_ID,
                client_secret=settings.GOOGLE_CLIENT_SECRET.get_secret_value(),
                authorization_endpoint="https://accounts.google.com/o/oauth2/v2/auth",
                token_endpoint="https://oauth2.googleapis.com/token",
                scope="openid email profile",
                redirect_uri=f"{base}/api/v1/auth/oauth/google/callback",
            )
        case "github":
            return OAuthProviderConfig(
                client_id=settings.GITHUB_CLIENT_ID,
                client_secret=settings.GITHUB_CLIENT_SECRET.get_secret_value(),
                authorization_endpoint="https://github.com/login/oauth/authorize",
                token_endpoint="https://github.com/login/oauth/access_token",
                scope="read:user user:email",
                redirect_uri=f"{base}/api/v1/auth/oauth/github/callback",
            )
        case _:
            msg = f"Unsupported OAuth provider: {provider}"
            raise ValidationException(msg)


class OAuthUserInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    email: str
    provider_user_id: str
    full_name: str | None = None
    email_verified: bool = False


async def fetch_oauth_userinfo(
    provider: str,
    config: OAuthProviderConfig,
    code: str,
) -> OAuthUserInfo:
    """Exchange authorization code for token, then fetch and normalize userinfo."""
    async with AsyncOAuth2Client(
        client_id=config.client_id,
        client_secret=config.client_secret,
    ) as client:
        fetch_kwargs: dict[str, Any] = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": config.redirect_uri,
        }
        # GitHub returns form-encoded by default; force JSON
        if provider == "github":
            fetch_kwargs["headers"] = {"Accept": "application/json"}

        await client.fetch_token(config.token_endpoint, **fetch_kwargs)

        if provider == "google":
            resp = await client.get("https://openidconnect.googleapis.com/v1/userinfo")
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, dict) or not data.get("email") or not data.get("sub"):
                msg = "OAuth provider returned an invalid identity"
                raise ValidationException(msg)
            return OAuthUserInfo(
                email=data["email"],
                provider_user_id=data["sub"],
                full_name=data.get("name"),
                email_verified=data.get("email_verified", False) is True,
            )

        # GitHub: primary email may not be public — requires separate emails endpoint
        resp = await client.get(
            "https://api.github.com/user",
            headers={"Accept": "application/vnd.github+json"},
        )
        resp.raise_for_status()
        user_data = resp.json()
        if not isinstance(user_data, dict) or not user_data.get("id"):
            msg = "OAuth provider returned an invalid identity"
            raise ValidationException(msg)
        emails_resp = await client.get(
            "https://api.github.com/user/emails",
            headers={"Accept": "application/vnd.github+json"},
        )
        emails_resp.raise_for_status()
        emails_payload = emails_resp.json()
        if not isinstance(emails_payload, list) or not all(
            isinstance(item, dict) for item in emails_payload
        ):
            msg = "GitHub returned an invalid email list"
            raise ValidationException(msg)
        email = next(
            (
                item["email"]
                for item in emails_payload
                if item.get("primary") and item.get("verified") and item.get("email")
            ),
            None,
        )

        if not email:
            msg = "GitHub account has no verified primary email"
            raise ValidationException(msg)

        return OAuthUserInfo(
            email=email,
            provider_user_id=str(user_data["id"]),
            full_name=user_data.get("name"),
            email_verified=True,
        )
