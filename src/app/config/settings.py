# src/settings.py

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
        default_factory=lambda: ["X-Total-Count", "X-Correlation-ID", "X-Process-Time", "Link"]
    )
    CORS_ALLOW_CREDENTIALS: bool = Field(default=True)
    CORS_MAX_AGE: int = Field(default=3600)

    # --- Server Configuration ---
    HOST: str = Field(default="0.0.0.0")  # noqa: S104
    PORT: int = Field(default=5000)
    WORKERS: int = Field(default=1)

    # --- MCP Configuration ---
    MCP_ENABLE_STDIO: bool = Field(default=True)
    MCP_ENABLE_HTTP: bool = Field(default=True)
    MCP_SERVER_NAME: str = Field(default="LangChain FastAPI MCP")
    MCP_HTTP_PATH: str = Field(default="/mcp")
    MCP_HTTP_TRANSPORT: str = Field(default="http")
    MCP_RUN_TRANSPORT: str = Field(default="stdio")
    MCP_HOST: str = Field(default="0.0.0.0")  # noqa: S104
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
    MCP_CLIENT_RETRY_ATTEMPTS: int = Field(default=1)
    MCP_CLIENT_CIRCUIT_BREAKER_THRESHOLD: int = Field(default=3)
    MCP_CLIENT_CIRCUIT_BREAKER_COOLDOWN_SECONDS: int = Field(default=60)

    # --- Mongo Database ---
    MONGODB_URI: str = Field(default="mongodb://localhost:27017/langchain_db")
    MONGODB_DB_NAME: str = Field(default="langchain_db")

    # --- PostgreSQL Database ---
    POSTGRES_URL: str = Field(
        default="postgresql://user:pass@host/db"
    )  # Added this missing field
    POSTGRES_MAX_OVERFLOW: int = Field(default=10)  # Added this missing field
    POSTGRES_POOL_SIZE: int = Field(default=5)  # Added this missing field

    # --- Neo4j Database ---
    NEO4J_URI: str = Field(default="bolt://localhost:7687")
    NEO4J_USERNAME: str = Field(default="neo4j")
    NEO4J_PASSWORD: str = Field(default="password")
    NEO4J_DATABASE: str = Field(default="neo4j")

    # --- Redis Cache ---
    REDIS_URL: str = Field(default="redis://localhost:6379")
    REDIS_HOST: str = Field(default="localhost")
    REDIS_PORT: int = Field(default=6379)
    REDIS_USERNAME: str = Field(default="default")
    REDIS_PASSWORD: str | None = Field(default=None)

    # Note: REDIS_DB and CACHE_TTL were not in your ENV, so they remain as defaults
    REDIS_DB: int = Field(default=0)
    CACHE_TTL: int = Field(default=3600)

    # ---RabbitMQ Configuration---
    RABBITMQ_URL: str = Field(default="amqp://guest:guest@localhost:5672//")
    RABBITMQ_PRIVATE_URL: str = Field(default="amqp://guest:guest@localhost:5672//")
    RABBITMQ_DEFAULT_USER: str = Field(default="guest")
    RABBITMQ_DEFAULT_PASS: str = Field(default="guest")
    RABBITMQ_NODENAME: str = Field(default="rabbit@localhost")
    CELERY_DEFAULT_QUEUE: str = Field(default="default")
    CELERY_DEFAULT_EXCHANGE: str = Field(default="tasks")
    CELERY_DEFAULT_ROUTING_KEY: str = Field(default="task.default")
    CELERY_DEAD_LETTER_EXCHANGE: str = Field(default="tasks.dlx")
    CELERY_DEAD_LETTER_QUEUE: str = Field(default="default.dlq")
    CELERY_DEAD_LETTER_ROUTING_KEY: str = Field(default="task.default.dlq")
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

    # --- Google Gemini API ---
    GOOGLE_API_KEY: str = Field(default="")
    GEMINI_MODEL: str = Field(default="gemini-2.5-flash")
    GEMINI_VISION_MODEL: str = Field(default="gemini-2.5-flash-image")
    GEMINI_EMBEDDING_MODEL: str = Field(default="text-embedding-005")
    GEMINI_TEMPERATURE: float = Field(default=0.7)
    GEMINI_MAX_TOKENS: int = Field(default=2048)
    LANGEXTRACT_API_KEY: str = Field(
        default="empty-langextract-api-key"
    )  # Added this missing field

    # --- Pinecone Vector Database ---
    PINECONE_API_KEY: str = Field(default="")
    PINECONE_ENVIRONMENT: str = Field(default="")
    PINECONE_INDEX_NAME: str = Field(default="langchain-index")
    PINECONE_DIMENSION: int = Field(default=768)
    PINECONE_METRIC: str = Field(default="cosine")

    # --- LangSmith ---
    # Renamed to match the variable in your ENV file: LANGSMITH_TRACING=true
    LANGSMITH_TRACING: bool = Field(default=False)
    LANGSMITH_ENDPOINT: str = Field(default="https://api.smith.langchain.com")
    LANGSMITH_API_KEY: str = Field(default="")  # Note: Your ENV had LANGSMITH_API_KEY
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

    # --- Tavily Search Configuration ---
    TAVILY_API_KEY: str = Field(default="")

    # --- PageIndex Configuration ---
    PAGEINDEX_API_KEY: str = Field(default="")

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
    FASTAPI_GUARD_IPINFO_TOKEN: str | None = Field(default=None)
    FASTAPI_GUARD_LOG_FORMAT: str = Field(default="text")

    # --- JWT Authentication ---
    JWT_SECRET_KEY: str = Field(default="super-secret-change-this-in-production")
    JWT_ISSUER: str = Field(default="your-app")
    JWT_ALGORITHM: str = Field(default="HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=15)
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=30)
    PASSWORD_RESET_EXPIRE_MINUTES: int = Field(default=30)

    # --- OAuth Configuration ---
    OAUTH_STATE_SECRET: str = Field(default="your-oauth-state-secret")
    GOOGLE_CLIENT_ID: str = Field(default="")
    GOOGLE_CLIENT_SECRET: str = Field(default="")
    GITHUB_CLIENT_ID: str = Field(default="")
    GITHUB_CLIENT_SECRET: str = Field(default="")

    # --- URLs ---
    BACKEND_URL: str = Field(default="http://localhost:5000")
    FRONTEND_URL: str = Field(default="http://localhost:3000")

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
    RESEND_API_KEY: str = Field(default="")
    RESEND_FROM_EMAIL: str = Field(default="noreply@yourdomain.com")
    RESEND_VERIFICATION_TEMPLATE_ID: str = Field(default="")
    RESEND_PASSWORD_RESET_TEMPLATE_ID: str = Field(default="")

    # --- S3 / R2 Storage ---
    S3_BUCKET_NAME: str = Field(default="")
    S3_ENDPOINT_URL: str | None = Field(default=None)
    S3_ACCESS_KEY_ID: str = Field(default="")
    S3_SECRET_ACCESS_KEY: str = Field(default="")
    S3_REGION: str = Field(default="auto")
    S3_PUBLIC_URL: str = Field(default="")

    # --- File Upload ---
    MAX_UPLOAD_SIZE: int = Field(default=10485760)  # 10MB
    ALLOWED_EXTENSIONS: list[str] = Field(
        default_factory=lambda: ["pdf", "txt", "docx", "xlsx", "pptx", "md", "html"]
    )

    # --- OpenTelemetry ---
    OTEL_EXPORTER_OTLP_ENDPOINT: str = Field(default="http://localhost:4317")
    OTEL_SERVICE_NAME: str = Field(default="langchain-fastapi")
    OTEL_TRACES_EXPORTER: str = Field(default="otlp")
    OTEL_METRICS_EXPORTER: str = Field(default="otlp")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Returns a cached instance of the application settings."""
    # Instantiating the class here ensures it's only done once (due to @lru_cache)
    return Settings()
