"""Application settings and configuration management."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",
    )

    # Application Settings
    app_name: str = Field(default="LangChain FastAPI Production")
    app_version: str = Field(default="1.0.0")
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    api_prefix: str = Field(default="/api/v1")
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])

    # Server Configuration
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=5000)
    workers: int = Field(default=4)

    # Security
    secret_key: str = Field(default="change-this-secret-key")
    jwt_algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=30)
    refresh_token_expire_days: int = Field(default=7)

    # Database
    mongodb_url: str = Field(default="mongodb://localhost:27017/langchain_db")
    mongodb_database: str = Field(default="langchain_db")

    # Redis Cache
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)
    redis_password: Optional[str] = Field(default=None)
    cache_ttl: int = Field(default=3600)

    # Google Gemini API
    google_api_key: str = Field(default="")
    gemini_model: str = Field(default="gemini-pro")
    gemini_vision_model: str = Field(default="gemini-pro-vision")
    gemini_embedding_model: str = Field(default="models/embedding-001")
    gemini_temperature: float = Field(default=0.7)
    gemini_max_tokens: int = Field(default=2048)

    # Pinecone Vector Database
    pinecone_api_key: str = Field(default="")
    pinecone_environment: str = Field(default="")
    pinecone_index_name: str = Field(default="langchain-index")
    pinecone_dimension: int = Field(default=768)
    pinecone_metric: str = Field(default="cosine")

    # LangSmith
    langsmith_api_key: str = Field(default="")
    langsmith_project: str = Field(default="langchain-production")
    langsmith_endpoint: str = Field(default="https://api.smith.langchain.com")
    langchain_tracing_v2: bool = Field(default=False)
    langchain_project: str = Field(default="langchain-production")

    # Crawl4AI Configuration
    crawl4ai_headless: bool = Field(default=True)
    crawl4ai_timeout: int = Field(default=30000)
    crawl4ai_user_agent: str = Field(
        default="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    )

    # Logging
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")
    log_file: str = Field(default="logs/app.log")

    # Rate Limiting
    rate_limit_enabled: bool = Field(default=True)
    rate_limit_requests: int = Field(default=100)
    rate_limit_period: int = Field(default=60)

    # File Upload
    max_upload_size: int = Field(default=10485760)  # 10MB
    allowed_extensions: List[str] = Field(
        default_factory=lambda: ["pdf", "txt", "docx", "xlsx", "pptx", "md", "html"]
    )

    # OpenTelemetry
    otel_exporter_otlp_endpoint: str = Field(default="http://localhost:4317")
    otel_service_name: str = Field(default="langchain-fastapi")
    otel_traces_exporter: str = Field(default="otlp")
    otel_metrics_exporter: str = Field(default="otlp")

    @field_validator("cors_origins", mode="before")
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            if v.startswith("["):
                return json.loads(v)
            return [origin.strip() for origin in v.split(",")]
        return v

    @field_validator("allowed_extensions", mode="before")
    def parse_allowed_extensions(cls, v):
        """Parse allowed extensions from string or list."""
        if isinstance(v, str):
            if v.startswith("["):
                return json.loads(v)
            return [ext.strip() for ext in v.split(",")]
        return v

    @property
    def redis_url(self) -> str:
        """Construct Redis URL."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment.lower() in ["development", "dev"]

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment.lower() in ["production", "prod"]

    @property
    def is_testing(self) -> bool:
        """Check if running in test mode."""
        return self.environment.lower() in ["testing", "test"]

    def get_langchain_env(self) -> Dict[str, Any]:
        """Get LangChain-specific environment variables."""
        return {
            "LANGCHAIN_TRACING_V2": str(self.langchain_tracing_v2).lower(),
            "LANGCHAIN_PROJECT": self.langchain_project,
            "LANGSMITH_API_KEY": self.langsmith_api_key,
            "LANGSMITH_ENDPOINT": self.langsmith_endpoint,
        }

    def validate_config(self) -> None:
        """Validate critical configuration."""
        errors = []

        if not self.google_api_key and not self.is_testing:
            errors.append("GOOGLE_API_KEY is required")

        if not self.pinecone_api_key and not self.is_testing:
            errors.append("PINECONE_API_KEY is required")

        if self.is_production:
            if self.secret_key == "change-this-secret-key":
                errors.append("SECRET_KEY must be changed in production")
            if self.debug:
                errors.append("DEBUG must be False in production")

        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()
    # Set LangChain environment variables
    import os
    for key, value in settings.get_langchain_env().items():
        os.environ[key] = str(value)
    return settings


# Create global settings instance
settings = get_settings()
