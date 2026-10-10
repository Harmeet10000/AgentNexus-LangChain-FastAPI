"""Explicit model-module registration for Alembic and metadata verification.

Every module below declares SQLAlchemy models that register tables on
`Base.metadata`. A missing import = a future autogenerate proposing DROP for
that table (the drop-safety contract). Order is load-bearing: payments before
subscriptions (cycle via subscriptions.dependencies). Adding a new model =
one string here.

The legacy shim keeps `chat_messages`/`document_vectors` on `Base.metadata`
until 0014-repaired tables are confirmed on all envs; remove after drop.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

MODEL_MODULE_NAMES: tuple[str, ...] = (
    "app.features.audit.model",
    "app.features.chat.model",
    "app.features.billing.credits.models.consumption",
    "app.features.billing.credits.models.credit",
    "app.features.documents.model",
    "app.features.billing.invoices.invoice_batch",
    "app.features.billing.invoices.invoice_void",
    "app.features.billing.invoices.model",
    "app.features.billing.invoices.receipt",
    "app.features.billing.invoices.report",
    "app.features.billing.payments.currency",
    "app.features.billing.payments.model",
    "app.features.billing.plans.model",
    "app.features.billing.subscriptions.model",
    "app.features.billing.subscriptions.trial_extension",
    "app.features.billing.webhooks.email_template",
    "app.features.billing.webhooks.model",
    "app.events.model",
    "database.schemas.document_vectors",
)


def register_models() -> tuple[ModuleType, ...]:
    """Import every model module for its registration side effect."""
    return tuple(importlib.import_module(name) for name in MODEL_MODULE_NAMES)
