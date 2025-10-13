"""Neon Postgres database configuration with SQLAlchemy."""
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from pydantic_settings import BaseSettings


class DatabaseConfig(BaseSettings):
    """Database configuration."""

    POSTGRES_URL: str  # Neon provides full connection string
    POSTGRES_POOL_SIZE: int = 20
    POSTGRES_MAX_OVERFLOW: int = 10

    @property
    def database_url(self) -> str:
        """Convert psycopg2 URL to asyncpg URL."""
        # Neon uses postgresql://, convert to postgresql+asyncpg://
        return self.POSTGRES_URL.replace("postgresql://", "postgresql+asyncpg://")

    class Config:
        env_file = ".env"


config = DatabaseConfig()

# Create async engine
engine = create_async_engine(
    config.database_url,
    echo=False,
    pool_size=config.POSTGRES_POOL_SIZE,
    max_overflow=config.POSTGRES_MAX_OVERFLOW,
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
