"""Pin the registered metadata set (restructure-domain-boundaries task 6.3).

`database.registry` is the single source of model-module registration. If a
module is dropped from it, its tables vanish from `Base.metadata` and the
next autogenerate proposes DROP — this test fails first, naming the missing
table rather than generating a destructive migration.
"""

from __future__ import annotations

from database.base import Base
from database.registry import MODEL_MODULE_NAMES, register_models

_EXPECTED_TABLES = frozenset(
    {
        "audit_logs",
        "chat_messages",
        "chat_sessions",
        "chunks",
        "credit_consumptions",
        "credit_notes",
        "credit_note_applications",
        "currencies",
        "dead_letter_events",
        "document_vectors",
        "documents",
        "email_templates",
        "fx_rates",
        "invoice_batches",
        "invoice_line_items",
        "invoice_voids",
        "invoices",
        "outbox_events",
        "payment_receipts",
        "provider_operation_records",
        "payments",
        "plans",
        "reports",
        "subscriptions",
        "app_tenants",
        "app_tenant_memberships",
        "trial_extensions",
        "user_credits",
        "webhook_events",
    }
)


def test_registry_registers_the_expected_table_set() -> None:
    modules = register_models()

    assert len(modules) == len(MODEL_MODULE_NAMES)
    assert frozenset(Base.metadata.tables.keys()) == _EXPECTED_TABLES
