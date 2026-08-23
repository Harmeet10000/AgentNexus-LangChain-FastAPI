# src/settings.py

import warnings
from decimal import Decimal
from functools import cache
from pathlib import Path
from typing import override

from loguru import logger
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PRODUCTION_SECRET_FIELDS: dict[str, list[str]] = {
    "JWT_SECRET_KEY": ["super-secret-change-this-in-production"],
    "NEO4J_PASSWORD": ["password"],
    "GEMINI_API_KEY": [""],
    "RESEND_API_KEY": [""],
    "OAUTH_STATE_SECRET": ["your-oauth-state-secret"],
    "S3_ACCESS_KEY_ID": [""],
    "S3_SECRET_ACCESS_KEY": [""],
    "TAVILY_API_KEY": [""],
    "PINECONE_API_KEY": [""],
    "RABBITMQ_DEFAULT_PASS": ["guest"],
    "POSTGRES_PASSWORD": ["pass"],
}


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Pydantic will now automatically look for the exact UPPERCASE name
    # defined in the class (e.g., Settings.APP_NAME will look for the APP_NAME env var).
    model_config = SettingsConfigDict(
        env_file=".env.development",
        env_file_encoding="utf-8",
        # Case sensitivity is set to True for clarity, but False works too
        # since we are enforcing the case in the model now.
        case_sensitive=True,
        slots=True,  # Faster attribute access  # ty:ignore[invalid-key]
        frozen=True,  # Immutable configuration
        extra="ignore",  # Prevents "pollution" from unknown env vars
    )

    @model_validator(mode="before")
    @classmethod
    def validate_embedding_dimension(cls, values: dict[str, object]) -> dict[str, object]:
        if "OTEL_ENABLED" not in values and values.get("ENVIRONMENT") == "production":
            values["OTEL_ENABLED"] = True

        dim = values.get("EMBEDDING_DIMENSION")
        model = values.get("GEMINI_EMBEDDING_MODEL", "")
        embedding_model_dimensions: dict[str, int] = {
            "gemini-embedding-2-preview": 768,
            "text-embedding-004": 768,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }
        if isinstance(model, str) and model in embedding_model_dimensions:
            expected = embedding_model_dimensions[model]
            if dim is not None and dim != expected:
                warnings.warn(
                    f"EMBEDDING_DIMENSION={dim} but {model} expects {expected}",
                    stacklevel=2,
                )
        return values

    # --- Application Settings ---
    APP_NAME: str = Field(default="LangChain FastAPI Production")
    APP_VERSION: str = Field(default="1.0.0")
    ENVIRONMENT: str = Field(default="development")
    API_PREFIX: str = Field(default="/api/v1")
    CORS_ORIGINS: list[str] = Field(default_factory=lambda: ["*"])
    CORS_ALLOW_METHODS: list[str] = Field(
        default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]
    )
    CORS_ALLOW_HEADERS: list[str] = Field(
        default_factory=lambda: [
            "Content-Type",
            "Authorization",
            "X-Correlation-ID",
            "Accept",
            "Cache-Control",
            "Connection",
        ]
    )
    CORS_EXPOSE_HEADERS: list[str] = Field(
        default_factory=lambda: [
            "X-Total-Count",
            "X-Correlation-ID",
            "X-Process-Time",
            "Link",
            "Deprecation",
            "Sunset",
        ]
    )
    CORS_ALLOW_CREDENTIALS: bool = Field(default=True)
    CORS_MAX_AGE: int = Field(default=3600)

    # --- Server Configuration ---
    HOST: str = Field(default="0.0.0.0")  # noqa: S104 — bind address, not a hardcoded secret

    # --- Server Configuration ---
    PORT: int = Field(default=5000)
    WORKERS: int = Field(default=1)

    # --- API Versioning ---
    API_SUNSET_DATE: str = Field(default="2027-06-15")
    API_V2_BASE_PATH: str = Field(default="/api/v2")

    # --- MCP Configuration ---
    MCP_ENABLE_STDIO: bool = Field(default=True)
    MCP_ENABLE_HTTP: bool = Field(default=True)
    MCP_SERVER_NAME: str = Field(default="LangChain FastAPI MCP")
    MCP_HTTP_PATH: str = Field(default="/mcp")
    MCP_HTTP_TRANSPORT: str = Field(default="http")
    MCP_RUN_TRANSPORT: str = Field(default="stdio")
    MCP_HOST: str = Field(default="0.0.0.0")  # noqa: S104 — bind address, not a hardcoded secret
    MCP_PORT: int = Field(default=8001)
    MCP_LOG_LEVEL: str = Field(default="INFO")
    MCP_REQUIRE_AUTH: bool = Field(default=True)
    MCP_REQUEST_TIMEOUT_SECONDS: int = Field(default=30)
    MCP_DEFAULT_PAGE_SIZE: int = Field(default=10)
    MCP_MAX_PAGE_SIZE: int = Field(default=50)
    MCP_MAX_RESULT_BYTES: int = Field(default=524288)
    MCP_SERVER_ENABLED_TOOLS: list[str] = Field(default_factory=list)
    MCP_RATE_LIMIT_BURST: int = Field(default=20)
    MCP_RATE_LIMIT_RATE: int = Field(default=20)
    MCP_RATE_LIMIT_PERIOD_SECONDS: int = Field(default=60)
    MCP_CLIENT_ENABLED: bool = Field(default=True)
    MCP_CLIENT_SERVER_CONFIGS: str = Field(default="[]")
    MCP_CLIENT_DEFAULT_TIMEOUT_SECONDS: int = Field(default=15)
    MCP_CLIENT_MAX_CONCURRENCY: int = Field(default=10)
    MCP_CLIENT_RETRY_ATTEMPTS: int = Field(default=3)
    MCP_CLIENT_CIRCUIT_BREAKER_THRESHOLD: int = Field(default=3)
    MCP_CLIENT_CIRCUIT_BREAKER_COOLDOWN_SECONDS: int = Field(default=60)

    # --- Mongo Database ---
    MONGODB_URI: str = Field(default="mongodb://localhost:27017/langchain_db")
    MONGODB_DB_NAME: str = Field(default="langchain_db")

    # --- PostgreSQL Database ---
    POSTGRES_URL: str = Field(default="postgresql://user:pass@host/db")  # Added this missing field
    POSTGRES_HOST: str = Field(default="localhost")
    POSTGRES_PORT: int = Field(default=5432)
    POSTGRES_USERNAME: str = Field(default="user")
    POSTGRES_PASSWORD: SecretStr = Field(default=SecretStr("pass"))
    POSTGRES_DB_NAME: str = Field(default="db")
    POSTGRES_MAX_OVERFLOW: int = Field(default=10)  # Added this missing field
    POSTGRES_POOL_SIZE: int = Field(default=5)  # Added this missing field

    # --- Neo4j Database ---
    NEO4J_URI: str = Field(default="bolt://localhost:7687")
    NEO4J_USERNAME: str = Field(default="neo4j")
    NEO4J_PASSWORD: SecretStr = Field(default=SecretStr("password"))
    NEO4J_DATABASE: str = Field(default="neo4j")

    # --- Redis Cache ---
    REDIS_URL: str = Field(default="redis://localhost:6379")
    REDIS_HOST: str = Field(default="localhost")
    REDIS_PORT: int = Field(default=6379)
    REDIS_USERNAME: str = Field(default="default")
    REDIS_PASSWORD: SecretStr | None = Field(default=None)

    # Note: REDIS_DB and CACHE_TTL were not in your ENV, so they remain as defaults
    REDIS_DB: int = Field(default=0)
    CACHE_TTL: int = Field(default=3600)

    # ---RabbitMQ Configuration---
    RABBITMQ_URL: str = Field(default="amqp://guest:guest@localhost:5672//")
    RABBITMQ_PRIVATE_URL: str = Field(default="amqp://guest:guest@localhost:5672//")
    RABBITMQ_DEFAULT_USER: str = Field(default="guest")
    RABBITMQ_DEFAULT_PASS: SecretStr = Field(default=SecretStr("guest"))
    RABBITMQ_NODENAME: str = Field(default="rabbit@localhost")
    CELERY_DEFAULT_QUEUE: str = Field(default="default")
    CELERY_DEFAULT_EXCHANGE: str = Field(default="tasks")
    CELERY_DEFAULT_ROUTING_KEY: str = Field(default="task.default")
    CELERY_DEAD_LETTER_EXCHANGE: str = Field(default="tasks.dlx")
    CELERY_DEAD_LETTER_QUEUE: str = Field(default="default.dlq")
    CELERY_DEAD_LETTER_ROUTING_KEY: str = Field(default="task.default.dlq")
    # Ingestion gets its own queue on the same task exchange, consumed by its own
    # worker service. Document ingestion is minutes of model work per message;
    # sharing one queue with sub-second billing and transactional-email tasks
    # means those wait behind it whenever every worker slot is busy. The
    # prefetch-of-one setting does not help: it stops one worker hoarding
    # messages, and does nothing about head-of-line blocking once every slot is
    # occupied. Two queues with disjoint consumers is what removes the coupling.
    CELERY_INGESTION_QUEUE: str = Field(default="ingestion")
    CELERY_INGESTION_ROUTING_KEY: str = Field(default="task.ingestion")
    CELERY_RETRY_MAX_RETRIES: int = Field(default=5)
    CELERY_RETRY_BACKOFF_MAX: int = Field(default=600)
    CELERY_DEFAULT_RETRY_DELAY: int = Field(default=5)
    CELERY_TASK_SOFT_TIME_LIMIT: int = Field(default=270)
    CELERY_TASK_TIME_LIMIT: int = Field(default=300)
    CELERY_TASK_RESULT_EXPIRES: int = Field(default=3600)
    CELERY_WORKER_MAX_TASKS_PER_CHILD: int = Field(default=500)
    CELERY_IDEMPOTENCY_TTL_SECONDS: int = Field(default=86400)
    CELERY_CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = Field(default=5)
    CELERY_CIRCUIT_BREAKER_RECOVERY_TIMEOUT: int = Field(default=60)

    # --- Credit System ---
    CREDIT_EXPIRATION_CRON_HOUR: int = Field(
        default=3, description="Hour of day for credit expiration job (0-23)"
    )
    CREDIT_EXPIRATION_CRON_MINUTE: int = Field(
        default=0, description="Minute of day for credit expiration job (0-59)"
    )
    CREDIT_RECONCILIATION_CRON_HOUR: int = Field(
        default=4, description="Hour of day for credit reconciliation job (0-23)"
    )
    CREDIT_RECONCILIATION_CRON_MINUTE: int = Field(
        default=0, description="Minute of day for credit reconciliation job (0-59)"
    )
    CREDIT_RECONCILIATION_CRON_DAY_OF_WEEK: str = Field(
        default="0", description="Day of week for reconciliation (0=Sunday)"
    )

    # --- Google Gemini API ---
    GEMINI_API_KEY: SecretStr = Field(default=SecretStr(""))
    GEMINI_FLASH_MODEL: str = Field(default="gemini-3.1-flash")
    GEMINI_PRO_MODEL: str = Field(default="gemini-3.1-pro")
    GEMINI_VISION_MODEL: str = Field(default="gemini-2.5-flash-image")
    GEMINI_EMBEDDING_MODEL: str = Field(default="gemini-embedding-2-preview")
    GEMINI_TEMPERATURE: float = Field(default=0.7, ge=0.0, le=2.0)
    GEMINI_TOP_P: float = Field(default=0.8, ge=0.0, le=1.0)
    GEMINI_TOP_K: int = Field(default=20, gt=0)
    GEMINI_MAX_TOKENS: int = Field(default=2048)
    GEMINI_CONTEXT_CACHE_TTL: str = Field(default="3600s", min_length=1)
    LANGEXTRACT_API_KEY: SecretStr = Field(
        default=SecretStr("empty-langextract-api-key")
    )  # Added this missing field

    # --- Pinecone Vector Database ---
    PINECONE_API_KEY: SecretStr = Field(default=SecretStr(""))
    PINECONE_ENVIRONMENT: str = Field(default="")
    PINECONE_INDEX_NAME: str = Field(default="langchain-index")
    PINECONE_DIMENSION: int = Field(default=768)
    PINECONE_METRIC: str = Field(default="cosine")

    # --- Embedding ---
    EMBEDDING_DIMENSION: int = Field(default=768, gt=0)

    # --- LangSmith ---
    # Renamed to match the variable in your ENV file: LANGSMITH_TRACING=true
    LANGSMITH_TRACING: bool = Field(default=False)
    LANGSMITH_ENDPOINT: str = Field(default="https://api.smith.langchain.com")
    LANGSMITH_API_KEY: SecretStr = Field(
        default=SecretStr("")
    )  # Note: Your ENV had LANGSMITH_API_KEY
    LANGSMITH_PROJECT: str = Field(default="langchain-production")
    LANGCHAIN_PROJECT: str = Field(
        default="langchain-production"
    )  # Note: Your ENV had LANGCHAIN_PROJECT

    # --- Crawl4AI Configuration ---
    CRAWL4AI_HEADLESS: bool = Field(default=True)
    CRAWL4AI_TIMEOUT: int = Field(default=30000)
    CRAWL4AI_USER_AGENT: str = Field(
        default="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    )
    CRAWL4AI_PROXY: str | None = Field(default=None)
    CRAWL4AI_PROXY_ENABLED: bool = Field(default=False)
    CRAWL4AI_MAX_DEPTH: int = Field(default=3)
    CRAWL4AI_MAX_PAGES: int = Field(default=10)
    CRAWL4AI_MAX_CONTENT_SIZE: int = Field(default=102400)  # 100KB
    CRAWL4AI_EXCLUDED_TAGS: str = Field(default="nav,footer,header,aside,form")
    CRAWL4AI_WORD_COUNT_THRESHOLD: int = Field(default=30)
    CRAWL4AI_PRUNING_THRESHOLD: float = Field(default=0.5)
    CRAWL4AI_PAGE_TIMEOUT: int = Field(default=30000)
    CRAWL4AI_WAIT_UNTIL: str | None = Field(default=None)
    CRAWL4AI_WAIT_FOR: str | None = Field(default=None)
    CRAWL4AI_VIEWPORT_WIDTH: int = Field(default=1280)
    CRAWL4AI_VIEWPORT_HEIGHT: int = Field(default=720)
    CRAWL4AI_STEALTH: bool = Field(default=False)
    CRAWL4AI_RATE_LIMIT_DELAY_MIN: float = Field(default=0.5)
    CRAWL4AI_RATE_LIMIT_DELAY_MAX: float = Field(default=1.0)

    # --- Crawl4AI: Magic mode ---
    CRAWL4AI_MAGIC: bool = Field(default=False)
    CRAWL4AI_LOCALE: str | None = Field(default=None)
    CRAWL4AI_TIMEZONE_ID: str | None = Field(default=None)
    CRAWL4AI_GEO_LAT: float | None = Field(default=None)
    CRAWL4AI_GEO_LON: float | None = Field(default=None)

    # --- Crawl4AI: Monitor ---
    CRAWL4AI_ENABLE_MONITOR: bool = Field(default=False)

    # --- Tavily Search Configuration ---
    TAVILY_API_KEY: SecretStr = Field(default=SecretStr(""))
    TAVILY_BASE_URL: str = Field(default="https://api.tavily.com")
    TAVILY_MAX_RESULTS_LIMIT: int = Field(default=20)
    TAVILY_TIMEOUT_SECONDS: float = Field(default=30.0)

    # --- PageIndex Configuration ---
    PAGEINDEX_API_KEY: SecretStr = Field(default=SecretStr(""))

    # --- Crawl/Search Rate Limiting ---
    CRAWL_RATE_LIMIT_PER_MINUTE: int = Field(default=10)
    CRAWL_RATE_LIMIT_PER_HOUR: int = Field(default=100)
    SEARCH_RATE_LIMIT_PER_MINUTE: int = Field(default=30)
    SEARCH_RATE_LIMIT_PER_HOUR: int = Field(default=500)

    # --- Redis Cache for Crawler ---
    REDIS_CRAWL_CACHE_TTL: int = Field(default=3600)  # 1 hour

    # --- Logging ---
    DEBUG: bool = Field(default=False)
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FORMAT: str = Field(default="json")
    LOG_FILE: str = Field(default="logs/app.log")
    LOG_BACKTRACE: bool = Field(default=True)
    LOG_DIAGNOSE: bool = Field(default=True)
    LOG_ROTATION: str = Field(default="10 MB")
    LOG_RETENTION: str = Field(default="7 days")
    LOG_COMPRESSION: str = Field(default="zip")
    LOG_DIR: Path = Path("logs")

    # --- Rate Limiting ---
    RATE_LIMIT_ENABLED: bool = Field(default=True)
    RATE_LIMIT_REQUESTS: int = Field(default=100)
    RATE_LIMIT_PERIOD: int = Field(default=60)

    # --- FastAPI Guard ---
    FASTAPI_GUARD_PASSIVE_MODE: bool | None = Field(default=None)
    FASTAPI_GUARD_ENFORCE_HTTPS: bool | None = Field(default=None)
    FASTAPI_GUARD_ENABLE_REDIS: bool = Field(default=True)
    FASTAPI_GUARD_TRUSTED_PROXIES: list[str] = Field(default_factory=list)
    FASTAPI_GUARD_TRUSTED_PROXY_DEPTH: int = Field(default=1)
    FASTAPI_GUARD_WHITELIST: list[str] | None = Field(default=None)
    FASTAPI_GUARD_BLACKLIST: list[str] = Field(default_factory=list)
    FASTAPI_GUARD_BLOCKED_USER_AGENTS: list[str] = Field(default_factory=list)
    FASTAPI_GUARD_AUTO_BAN_THRESHOLD: int = Field(default=10)
    FASTAPI_GUARD_AUTO_BAN_DURATION: int = Field(default=3600)
    FASTAPI_GUARD_BLOCKED_COUNTRIES: list[str] = Field(default_factory=list)
    FASTAPI_GUARD_WHITELIST_COUNTRIES: list[str] = Field(default_factory=list)
    FASTAPI_GUARD_BLOCK_CLOUD_PROVIDERS: list[str] = Field(default_factory=list)
    FASTAPI_GUARD_IPINFO_TOKEN: SecretStr | None = Field(default=None)
    FASTAPI_GUARD_LOG_FORMAT: str = Field(default="text")

    # --- JWT Authentication ---
    JWT_SECRET_KEY: SecretStr = Field(default=SecretStr("super-secret-change-this-in-production"))
    JWT_ISSUER: str = Field(default="your-app")
    JWT_ALGORITHM: str = Field(default="HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=15)
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=30)
    PASSWORD_RESET_EXPIRE_MINUTES: int = Field(default=30)

    # --- OAuth Configuration ---
    OAUTH_STATE_SECRET: SecretStr = Field(default=SecretStr("your-oauth-state-secret"))
    GOOGLE_CLIENT_ID: str = Field(default="")
    GOOGLE_CLIENT_SECRET: SecretStr = Field(default=SecretStr(""))
    GITHUB_CLIENT_ID: str = Field(default="")
    GITHUB_CLIENT_SECRET: SecretStr = Field(default=SecretStr(""))

    # --- URLs ---
    BACKEND_URL: str = Field(default="http://localhost:5000")
    FRONTEND_URL: str = Field(default="http://localhost:3000")
    RESEND_SEND_URL: str = Field(default="https://api.resend.com/emails")

    # --- WebSocket Security ---
    WEBSOCKET_ALLOWED_ORIGINS: list[str] = Field(default_factory=list)
    WEBSOCKET_REQUIRE_ORIGIN: bool = Field(default=True)
    WEBSOCKET_IDLE_TIMEOUT_SECONDS: int = Field(default=180)
    WEBSOCKET_USER_MESSAGE_RATE: int = Field(default=60)
    WEBSOCKET_USER_MESSAGE_PERIOD_SECONDS: int = Field(default=60)
    WEBSOCKET_CONNECTION_MESSAGE_RATE: int = Field(default=20)
    WEBSOCKET_CONNECTION_MESSAGE_PERIOD_SECONDS: int = Field(default=10)
    WEBSOCKET_MAX_CONNECTIONS_PER_USER: int = Field(default=3)
    WEBSOCKET_PRESENCE_TTL_SECONDS: int = Field(default=360)

    # --- Uvicorn WebSocket Transport ---
    UVICORN_WS_MAX_SIZE: int = Field(default=1_048_576)
    UVICORN_WS_MAX_QUEUE: int = Field(default=32)
    UVICORN_WS_PING_INTERVAL: float = Field(default=20.0)
    UVICORN_WS_PING_TIMEOUT: float = Field(default=20.0)

    # --- Resend Email Service ---
    RESEND_API_KEY: SecretStr = Field(default=SecretStr(""))
    RESEND_FROM_EMAIL: str = Field(default="noreply@yourdomain.com")
    RESEND_VERIFICATION_TEMPLATE_ID: str = Field(default="")
    RESEND_PASSWORD_RESET_TEMPLATE_ID: str = Field(default="")

    # --- S3 / R2 Storage ---
    S3_BUCKET_NAME: str = Field(default="")
    S3_ENDPOINT_URL: str | None = Field(default=None)
    S3_ACCESS_KEY_ID: SecretStr = Field(default=SecretStr(""))
    S3_SECRET_ACCESS_KEY: SecretStr = Field(default=SecretStr(""))
    S3_REGION: str = Field(default="auto")
    S3_FORCE_PATH_STYLE: bool = Field(default=True)
    S3_PUBLIC_URL: str = Field(default="")

    # --- File Upload ---
    MAX_UPLOAD_SIZE: int = Field(default=10485760)  # 10MB
    ALLOWED_EXTENSIONS: list[str] = Field(
        default_factory=lambda: ["pdf", "txt", "docx", "xlsx", "pptx", "md", "html"]
    )

    # --- Razorpay Billing ---
    RAZORPAY_KEY_ID: str = Field(default="")
    RAZORPAY_KEY_SECRET: SecretStr = Field(default=SecretStr(""))
    RAZORPAY_WEBHOOK_SECRET: SecretStr = Field(default=SecretStr(""))
    RAZORPAY_API_BASE_URL: str = Field(default="https://api.razorpay.com")
    RAZORPAY_REQUEST_TIMEOUT_SECONDS: float = Field(default=15.0)
    # GST configuration
    BILLING_SELLER_GSTIN: str = Field(default="")
    BILLING_SELLER_STATE_CODE: str = Field(default="27")
    BILLING_PLACE_OF_SUPPLY: str = Field(default="27")
    BILLING_DEFAULT_TAX_RATE: Decimal = Field(default=Decimal("0.18"))
    BILLING_INVOICE_PREFIX: str = Field(default="INV")
    BILLING_RECEIPT_PREFIX: str = Field(default="REC")
    BILLING_DUNNING_RETRY_DAYS: list[int] = Field(default_factory=lambda: [1, 3, 7, 14])
    BILLING_MAX_RETRIES: int = Field(default=4)
    BILLING_RECONCILIATION_LOOKBACK_DAYS: int = Field(default=7)

    # --- OpenTelemetry ---
    OTEL_EXPORTER_OTLP_ENDPOINT: str = Field(default="http://localhost:4317")
    OTEL_SERVICE_NAME: str = Field(default="langchain-fastapi")
    OTEL_TRACES_EXPORTER: str = Field(default="otlp")
    OTEL_METRICS_EXPORTER: str = Field(default="otlp")
    OTEL_LOGS_EXPORTER: str = Field(default="otlp")
    OTEL_ENABLED: bool = Field(default=False)
    OTEL_SAMPLE_RATE: float = Field(default=1.0, ge=0.0, le=1.0)

    @override
    def model_post_init(self, __context: object) -> None:
        bad_fields: list[str] = []
        for field_name, bad_defaults in PRODUCTION_SECRET_FIELDS.items():
            raw = getattr(self, field_name, None)
            if raw is None:
                continue
            value = raw.get_secret_value() if isinstance(raw, SecretStr) else str(raw)
            if value in bad_defaults:
                bad_fields.append(field_name)

        if self.ENVIRONMENT == "production" and bad_fields:
            error_lines = "\n".join(f"  - {f}" for f in bad_fields)
            msg = (
                "Settings validation failed for production environment.\n"
                "The following secret fields have default/insecure values:\n"
                f"{error_lines}\n"
                "Set these environment variables before starting the application."
            )
            raise ValueError(msg)

        if self.ENVIRONMENT != "production" and bad_fields:
            logger.warning(
                "Secret fields have default values (safe in {}): {}",
                self.ENVIRONMENT,
                ", ".join(bad_fields),
            )


@cache
def get_settings() -> Settings:
    """Returns a cached instance of the application settings."""
    return Settings()
