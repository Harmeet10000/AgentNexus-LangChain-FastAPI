"""Neon Postgres database configuration with SQLAlchemy."""

from collections.abc import AsyncGenerator
from urllib.parse import urlparse

from fastapi import Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config.settings import get_settings
from app.utils.logger import logger

settings = get_settings()


def get_database_url() -> str:
    """Convert psycopg2 URL to asyncpg URL."""
    postgres_url = settings.POSTGRES_URL
    asyncpg_url = postgres_url.replace("postgresql://", "postgresql+asyncpg://")
    asyncpg_url = asyncpg_url.replace("&sslmode=require", "")
    asyncpg_url = asyncpg_url.replace("&channel_binding=require", "")
    asyncpg_url = asyncpg_url.replace("?sslmode=require", "")
    asyncpg_url = asyncpg_url.replace("?channel_binding=require", "")
    return asyncpg_url


async def init_db() -> tuple[create_async_engine, async_sessionmaker[AsyncSession]]:
    """Initialize database engine and session factory.

    Returns:
        tuple: (engine, AsyncSessionLocal) for app.state injection
    """
    engine = create_async_engine(
        url=get_database_url(),
        echo=False,
        pool_size=settings.POSTGRES_POOL_SIZE,
        max_overflow=settings.POSTGRES_MAX_OVERFLOW,
        pool_pre_ping=True,
        pool_timeout=30,
        pool_recycle=3600,
    )

    session_local = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )

    # Test connection and log version info
    try:
        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT version()"))
            version = result.scalar()
            parsed_url = urlparse(settings.POSTGRES_URL)
            host = parsed_url.hostname

            logger.info(
                "PostgreSQL connected",
                host=host,
                database=settings.POSTGRES_URL.split("/")[-1],
                version=version,
            )
    except Exception as e:
        logger.error(f"PostgreSQL initialization failed: {e}", exc_info=True)
        await engine.dispose()
        raise

    return engine, session_local


async def get_postgres_db(request: Request) -> AsyncGenerator[AsyncSession, None]:
    """Dependency for database sessions retrieved from app.state."""
    session_local = request.app.state.db_session_local
    async with session_local() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
