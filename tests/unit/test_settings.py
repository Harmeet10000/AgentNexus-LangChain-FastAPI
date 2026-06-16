import re

import pytest

from app.config.settings import Settings


class TestSettingsProductionValidation:
    def test_raises_value_error_with_bad_secrets(self) -> None:
        with pytest.raises(ValueError, match="production environment"):
            Settings(ENVIRONMENT="production")

    def test_succeeds_with_proper_secrets(self) -> None:
        settings = Settings(
            ENVIRONMENT="production",
            JWT_SECRET_KEY="a-very-strong-random-secret-32chars!!",
            NEO4J_PASSWORD="real-password-123",
            GEMINI_API_KEY="valid-key",
            RESEND_API_KEY="valid-key",
            OAUTH_STATE_SECRET="valid-secret",
            S3_ACCESS_KEY_ID="valid-id",
            S3_SECRET_ACCESS_KEY="valid-key",
            TAVILY_API_KEY="valid-key",
            PINECONE_API_KEY="valid-key",
        )
        assert settings.ENVIRONMENT == "production"

    def test_succeeds_in_development_with_bad_secrets(self) -> None:
        settings = Settings(ENVIRONMENT="development")
        assert settings.ENVIRONMENT == "development"

    def test_error_lists_all_bad_field_names(self) -> None:
        with pytest.raises(ValueError, match="The following secret fields have default/insecure values"):
            Settings(ENVIRONMENT="production")

    def test_error_does_not_expose_secret_values(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            Settings(ENVIRONMENT="production")
        message = str(exc_info.value)
        assert "super-secret-change-this-in-production" not in message
        assert "password" not in message.split("\n")[-1]
