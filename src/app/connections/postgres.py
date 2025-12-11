"""Neon Postgres database configuration with SQLAlchemy."""

from collections.abc import AsyncGenerator
from urllib.parse import urlparse

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.settings import get_settings
from app.utils.logger import logger


def get_database_url() -> str:
    """Convert psycopg2 URL to asyncpg URL."""
    postgres_url = get_settings().POSTGRES_URL
    # Convert to asyncpg and fix SSL parameters
    asyncpg_url = postgres_url.replace("postgresql://", "postgresql+asyncpg://")
    # Remove psycopg2-specific SSL parameters that asyncpg doesn't support
    asyncpg_url = asyncpg_url.replace("&sslmode=require", "")
    asyncpg_url = asyncpg_url.replace("&channel_binding=require", "")
    asyncpg_url = asyncpg_url.replace("?sslmode=require", "")
    asyncpg_url = asyncpg_url.replace("?channel_binding=require", "")
    return asyncpg_url


# Create async engine
engine = create_async_engine(
    get_database_url(),
    echo=False,
    pool_size=get_settings().POSTGRES_POOL_SIZE,
    max_overflow=get_settings().POSTGRES_MAX_OVERFLOW,
    pool_pre_ping=True,
)

# Create session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for database sessions."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Initialize database tables."""
    async with engine.begin() as conn:
        # Get connection info for logging
        result = await conn.execute(text("SELECT version()"))
        version = result.scalar()

        # Parse URL for host info
        parsed_url = urlparse(get_settings().POSTGRES_URL)
        host = parsed_url.hostname

        logger.info(
            f"PostgreSQL Connected: {host}",
            readyState=1,
            database="neon-postgres",
            version=version,
        )


async def close_db() -> None:
    """Close database connections."""
    await engine.dispose()
