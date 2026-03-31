"""Neon Postgres database configuration with SQLAlchemy."""
from collections.abc import AsyncGenerator
from urllib.parse import urlparse

from fastapi.requests import HTTPConnection

# from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import get_settings
from app.utils import logger

settings = get_settings()


def get_database_url() -> str:
    """Convert psycopg2 URL to asyncpg URL."""
    postgres_url = settings.POSTGRES_URL

    if postgres_url.startswith("postgresql+asyncpg://"):
        asyncpg_url = postgres_url
    elif postgres_url.startswith("postgresql://"):
        asyncpg_url = postgres_url.replace(
            "postgresql://",
            "postgresql+asyncpg://",
            1,
        )
    elif postgres_url.startswith("postgres://"):
        asyncpg_url = postgres_url.replace(
            "postgres://",
            "postgresql+asyncpg://",
            1,
        )
    else:
        asyncpg_url = postgres_url

    asyncpg_url = asyncpg_url.replace("&sslmode=require", "")
    asyncpg_url = asyncpg_url.replace("&channel_binding=require", "")
    asyncpg_url = asyncpg_url.replace("?sslmode=require", "")
    return asyncpg_url.replace("?channel_binding=require", "")


async def init_db() -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
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

    # Instrument SQLAlchemy engine for distributed tracing
    # SQLAlchemyInstrumentor().instrument(engine=engine)
    # logger.info("SQLAlchemy instrumentation enabled for OpenTelemetry")

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


async def get_postgres_db(connection: HTTPConnection) -> AsyncGenerator[AsyncSession, None]:
    """Dependency for database sessions retrieved from app.state."""
    session_local = connection.app.state.db_session_local
    async with session_local() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
