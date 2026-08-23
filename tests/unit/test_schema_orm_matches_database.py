"""The live-database half of the schema identifier work (band D14).

Deselected by default: ``addopts`` carries ``-m "not integration and not requires_db"``.
It is expected to be deselected in every current environment and is not expected to
pass until the database has been rebuilt by ``upgrade`` rather than ``stamp`` --
recorded in ``design.md``'s Risks section.

Read-only only: three catalog SELECTs, no writes, no DDL, no autogenerate comparison.
The fixture prints host/port/database and nothing else -- never a credential, never a
connection string.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

pytestmark = pytest.mark.requires_db

# Every model module must be imported before ``Base.metadata`` is read; this mirrors
# the side-effect import block in src/alembic/env.py.
_MODEL_MODULES = (
    "app.features.audit.model",
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
)


@pytest.fixture(scope="module")
def engine() -> Engine:
    import importlib

    import psycopg
    from sqlalchemy import create_engine

    for module_name in _MODEL_MODULES:
        importlib.import_module(module_name)

    from app.connections.postgres import get_database_url

    url = create_engine(get_database_url(flavour="plain")).url
    print(f"target: host={url.host} port={url.port} database={url.database}")  # noqa: T201
    return create_engine(
        get_database_url(flavour="plain"),
        connect_args={"connector": psycopg},  # driver already installed; explicit on purpose
    )


def _scalar_set(engine: Engine, query: str) -> set[tuple[str, ...]]:
    from sqlalchemy import text

    with engine.connect() as conn:
        return {tuple(row) for row in conn.execute(text(query))}


def test_every_declared_table_exists(engine: Engine) -> None:
    from database import Base

    declared = {t for t in Base.metadata.tables}
    existing = {
        row[0]
        for row in _scalar_set(
            engine,
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'",
        )
    }
    assert declared <= existing, f"tables declared in the ORM but absent from the database: {sorted(declared - existing)}"


def test_every_declared_column_exists(engine: Engine) -> None:
    from database import Base

    declared = {
        (table.name, column.name)
        for table in Base.metadata.tables.values()
        for column in table.columns
    }
    existing = _scalar_set(
        engine,
        "SELECT table_name, column_name FROM information_schema.columns "
        "WHERE table_schema = 'public'",
    )
    missing = sorted(declared - existing)
    assert not missing, f"columns declared in the ORM but absent from the database: {missing}"


def test_every_named_index_exists(engine: Engine) -> None:
    from sqlalchemy import Index

    from database import Base

    declared = {
        index.name
        for table in Base.metadata.tables.values()
        for index in table.indexes
        if isinstance(index, Index) and index.name is not None
    }
    existing = {
        row[0]
        for row in _scalar_set(
            engine,
            "SELECT indexname FROM pg_indexes WHERE schemaname = 'public'",
        )
    }
    assert declared <= existing, f"indexes declared in the ORM but absent from pg_indexes: {sorted(declared - existing)}"
