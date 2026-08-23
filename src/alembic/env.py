import asyncio
from logging.config import fileConfig

from sqlalchemy.engine import Connection

import app.features.audit.model
import app.features.credits.models.consumption
import app.features.credits.models.credit
import app.features.documents.model
import app.features.invoices.invoice_batch
import app.features.invoices.invoice_void
import app.features.invoices.model
import app.features.invoices.receipt
import app.features.invoices.report
import app.features.payments.currency
import app.features.payments.model
import app.features.plans.model
import app.features.subscriptions.model
import app.features.subscriptions.trial_extension
import app.features.webhooks.email_template
import app.features.webhooks.model
import app.shared.outbox.model
from alembic import context
from app.connections import init_db
from app.utils import logger
from database import Base

# Every module above is imported for one side effect: executing its body declares its
# SQLAlchemy models, which registers their tables on ``Base.metadata`` — the registry
# Alembic compares the database against. A table missing from that registry is a table a
# future comparison proposes to DROP, so the import list is the drop-safety contract and
# nothing in this file calls into it.
#
# The modules are therefore named a second time below rather than carrying a suppression.
# ``F401`` is in this project's ruff ``fixable`` set, so a bare side-effect import here is
# not merely flagged: ``ruff check --fix`` deletes it, silently unregistering the tables it
# was protecting. Referencing them removes the diagnostic instead of hiding it.
#
# Import order is load-bearing and alphabetical order happens to satisfy it:
# ``app.features.payments.model`` must precede ``app.features.subscriptions.model``,
# because the subscriptions package reaches payments and back through
# ``subscriptions.dependencies``. Entering that cycle from subscriptions raises
# ``ImportError``; entering it from payments does not.
#
# The two credit modules are registered for the same drop-safety reason as the rest, even
# though they are named nowhere in this change's task text. ``0005`` creates ``user_credits``
# and ``credit_consumptions``, and joining the heads put ``0005`` inside the single-head
# chain — so leaving them unregistered would have a future comparison propose dropping two
# relations the chain creates, which is precisely the defect this import list exists to
# prevent.
_MODEL_MODULES = (
    app.features.audit.model,
    app.features.credits.models.consumption,
    app.features.credits.models.credit,
    app.features.documents.model,
    app.features.invoices.invoice_batch,
    app.features.invoices.invoice_void,
    app.features.invoices.model,
    app.features.invoices.receipt,
    app.features.invoices.report,
    app.features.payments.currency,
    app.features.payments.model,
    app.features.plans.model,
    app.features.subscriptions.model,
    app.features.subscriptions.trial_extension,
    app.features.webhooks.email_template,
    app.features.webhooks.model,
    app.shared.outbox.model,
)

# Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# The agent-memory library creates its own tables inside a dedicated schema. They are
# the library's business, not this migration chain's: autogenerate must never propose
# creating or dropping them, because their lifecycle belongs to whoever owns the schema.
# The filter is wired into BOTH configure calls below — protecting only the online
# branch would leave `--autogenerate` against a connection free to emit memory-schema
# DDL into a revision file nobody reviews that closely.
MEMORY_SCHEMA_NAME = "cognee_memory"


def exclude_non_app_schema(obj: object, name: str, type_: str) -> bool:  # noqa: ARG001 — alembic callback signature
    """``include_object`` predicate: keep app tables, drop everything foreign.

    An object living in the memory schema (or any other non-default schema) is
    excluded; an ordinary table in ``Base.metadata`` passes through.
    """
    schema = getattr(obj, "schema", None)
    return not (schema is not None and schema != "public")


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
        include_object=exclude_non_app_schema,
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
        include_object=exclude_non_app_schema,
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
