"""Neon Postgres database configuration with SQLAlchemy."""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

from app.core.settings import get_settings


def get_database_url() -> str:
    """Convert psycopg2 URL to asyncpg URL."""
    postgres_url = get_settings().POSTGRES_URL
    return postgres_url.replace("postgresql://", "postgresql+asyncpg://")


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

# Base class for models
Base = declarative_base()


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
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Close database connections."""
    await engine.dispose()
