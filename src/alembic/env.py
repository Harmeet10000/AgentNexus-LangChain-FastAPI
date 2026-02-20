import asyncio
from logging.config import fileConfig

from sqlalchemy.engine import Connection

from alembic import context

# Import database initialization function
from app.connections.postgres import init_db
from app.utils.logger import logger

# Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Import all models here for autogenerate support
try:
    from database import Base

    # Import all models to register with Base.metadata
    from database.schemas import ChatMessage, ChatSession, DocumentVector

    target_metadata = Base.metadata
except ImportError as e:
    logger.warning(f"Failed to import models: {e}")
    target_metadata = None


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This is used for generating migration scripts without connecting to the database.
    """
    # For offline mode, we need a dummy URL or skip URL-based configuration
    url = config.get_main_option("sqlalchemy.url")
    if not url:
        # Fallback: construct a dummy URL
        url = "postgresql+asyncpg://localhost/dummy"

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Run migrations with the provided database connection.

    Args:
        connection: Active database connection from the engine
    """
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        render_as_batch=True,  # For compatibility with certain dialects
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in async mode using init_db() to get the engine."""
    try:
        logger.info("Starting database migrations")

        # Use init_db() to get the configured engine
        engine, _ = await init_db()

        logger.info("Database engine initialized for migrations")

        async with engine.connect() as connection:
            await connection.run_sync(do_run_migrations)
            logger.info("Database migrations completed successfully")

        await engine.dispose()
        logger.info("Database engine disposed")

    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        raise


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    This connects to the database and applies pending migrations.
    """
    try:
        asyncio.run(run_async_migrations())
    except Exception as e:
        logger.error(f"Failed to run migrations: {e}", exc_info=True)
        raise


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
