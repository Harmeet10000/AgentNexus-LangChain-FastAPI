"""Neon Postgres database configuration with SQLAlchemy."""

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING
from urllib.parse import urlparse, urlunparse

from fastapi.requests import HTTPConnection
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import get_settings
from app.config.settings import Settings
from app.utils import logger

if TYPE_CHECKING:
    from typing import Any
    from urllib.parse import ParseResult

    from sqlalchemy.engine.cursor import CursorResult
    from ty_extensions import Unknown

settings: Settings = get_settings()


def get_database_url() -> str:
    """Convert psycopg2 URL to asyncpg URL."""
    postgres_url: str = settings.POSTGRES_URL

    if postgres_url.startswith("postgresql+asyncpg://"):
        asyncpg_url: str = postgres_url
    elif postgres_url.startswith("postgresql://"):
        asyncpg_url: str = postgres_url.replace(
            "postgresql://",
            "postgresql+asyncpg://",
            1,
        )
    elif postgres_url.startswith("postgres://"):
        asyncpg_url: str = postgres_url.replace(
            "postgres://",
            "postgresql+asyncpg://",
            1,
        )
    else:
        asyncpg_url: str = postgres_url

    asyncpg_url: str = asyncpg_url.replace("&sslmode=require", "")
    asyncpg_url: str = asyncpg_url.replace("&channel_binding=require", "")
    asyncpg_url: str = asyncpg_url.replace("?sslmode=require", "")
    asyncpg_url = asyncpg_url.replace("?channel_binding=require", "")

    parsed: ParseResult = urlparse(asyncpg_url)
    if not parsed.password and settings.POSTGRES_PASSWORD.get_secret_value() != "pass":
        password: str = settings.POSTGRES_PASSWORD.get_secret_value()
        netloc: str = (
            f"{parsed.username}:{password}@{parsed.hostname}" if parsed.username else parsed.netloc
        )
        if parsed.port:
            netloc += f":{parsed.port}"
        asyncpg_url = urlunparse(
            (parsed.scheme, netloc, parsed.path, parsed.params, parsed.query, parsed.fragment)
        )
    return asyncpg_url


async def init_db() -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    """Initialize database engine and session factory.

    Returns:
        tuple: (engine, AsyncSessionLocal) for app.state injection
    """
    engine: AsyncEngine = create_async_engine(
        url=get_database_url(),
        echo=False,
        pool_size=settings.POSTGRES_POOL_SIZE,
        max_overflow=settings.POSTGRES_MAX_OVERFLOW,
        pool_pre_ping=True,
        pool_timeout=30,
        pool_recycle=3600,
        connect_args={
            # Timeouts are critical when managing your own pool directly
            "server_settings": {
                "statement_timeout": "10000",
                "idle_in_transaction_session_timeout": "10000",
            }
        },
        # if using Pg-Bouncer, set poolclass to NullPool to disable connection pooling at the SQLAlchemy level
        # Disable SQLAlchemy's pool, let PgBouncer handle concurrency.
        # Alternatively, keep poolclass default but set pool_size to something very small (e.g., 2 to 5).
        # poolclass=NullPool,
        # connect_args={
        #     # CRITICAL: Disables prepared statements to prevent PgBouncer transaction errors
        #     "prepared_statement_cache_size": 0,
        #     "statement_cache_size": 0,
        #     # Prevent bad queries from hanging connections forever (value in milliseconds)
        #     "server_settings": {
        #         "statement_timeout": "10000",
        #         "idle_in_transaction_session_timeout": "10000"
        #     }
        # }
    )

    session_local: async_sessionmaker[AsyncSession] = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )

    try:
        await _verify_postgres_connection(engine)
    except Exception as e:
        logger.error("PostgreSQL initialization failed: {}", e, exc_info=True)
        await engine.dispose()
        raise

    return engine, session_local


async def _verify_postgres_connection(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        result: CursorResult[Any] = await conn.execute(text("SELECT version()"))
        version: Unknown = result.scalar()
        parsed_url: ParseResult = urlparse(settings.POSTGRES_URL)
        host: str | None = parsed_url.hostname
        logger.info(
            "PostgreSQL connected",
            host=host,
            database=settings.POSTGRES_URL.split("/")[-1],
            version=version,
        )


async def get_postgres_db(connection: HTTPConnection) -> AsyncGenerator[AsyncSession, None]:
    """Dependency for database sessions retrieved from app.state."""
    session_local: Any = connection.app.state.db_session_local
    async with session_local() as session:
        try:
            yield session  # noqa: ASYNC119
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
