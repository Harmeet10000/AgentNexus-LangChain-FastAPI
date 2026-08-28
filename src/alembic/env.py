import asyncio
import importlib
from logging.config import fileConfig

from sqlalchemy.engine import Connection

from alembic import context
from app.connections import init_db
from app.utils import logger
from database import Base

# ---------------------------------------------------------------------------
# Drop-safety contract: every module below declares SQLAlchemy models that
# register tables on Base.metadata. A missing import = a future autogenerate
# proposing DROP for that table.
# ---------------------------------------------------------------------------
# ponytail: data-driven import so the list is one source of truth, order is
# still load-bearing (payments before subscriptions due to cycle via
# subscriptions.dependencies). Adding a new model = one string here.
_MODEL_MODULE_NAMES: tuple[str, ...] = (
    "app.features.audit.model",
    "app.features.chat.model",
    "app.features.credits.models.consumption",
    "app.features.credits.models.credit",
    "app.features.documents.model",
    "app.features.invoices.invoice_batch",
    "app.features.invoices.invoice_void",
    "app.features.invoices.model",
    "app.features.invoices.receipt",
    "app.features.invoices.report",
    "app.features.payments.currency",
    "app.features.payments.model",
    "app.features.plans.model",
    "app.features.subscriptions.model",
    "app.features.subscriptions.trial_extension",
    "app.features.webhooks.email_template",
    "app.features.webhooks.model",
    "app.shared.outbox.model",
    # legacy shim — keeps chat_messages/document_vectors on Base.metadata
    # until 0014-repaired tables are confirmed on all envs; remove after drop.
    "database.schemas.document_vectors",
)

# Side-effect imports — each executes its module body and registers tables.
# Ty sees these as used via _MODEL_MODULES, so F401 fix does not delete them.
_MODEL_MODULES = tuple(importlib.import_module(name) for name in _MODEL_MODULE_NAMES)

# Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# Cognee owns its own schema — never let autogenerate touch it.
MEMORY_SCHEMA_NAME = "cognee_memory"


def _is_memory_schema(schema: str | None) -> bool:
    return schema is not None and schema != "public" and schema == MEMORY_SCHEMA_NAME


def include_object(
    obj: object, name: str, type_: str, reflected: bool, compare_to: object | None
) -> bool:
    """Alembic include_object — exclude cognee_memory and non-public schemas."""
    schema = getattr(obj, "schema", None)
    if _is_memory_schema(schema):
        return False
    # reflected=True means object came from DB inspection; compare_to=None means new
    # object in metadata not yet in DB — both should respect same filter.
    return not (schema is not None and schema != "public")


def include_name(name: str | None, type_: str, parent_names: dict[str, str | None]) -> bool:
    """Alembic include_name — filter schema names themselves."""
    return not (type_ == "schema" and name == MEMORY_SCHEMA_NAME)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    if not url:
        url = "postgresql+asyncpg://localhost/dummy"

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        include_object=include_object,
        include_name=include_name,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Run migrations with the provided database connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        render_as_batch=True,
        include_object=include_object,
        include_name=include_name,
    )

    with context.begin_transaction():
        context.run_migrations()


async def _run_migrations() -> None:
    logger.info("Starting database migrations")
    engine, _ = await init_db()
    logger.info("Database engine initialized for migrations")
    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations)
        logger.info("Database migrations completed successfully")
    await engine.dispose()
    logger.info("Database engine disposed")


async def run_async_migrations() -> None:
    """Run migrations in async mode using init_db() to get the engine."""
    try:
        await _run_migrations()
    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        raise


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    try:
        asyncio.run(run_async_migrations())
    except Exception as e:
        logger.error(f"Failed to run migrations: {e}", exc_info=True)
        raise


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
