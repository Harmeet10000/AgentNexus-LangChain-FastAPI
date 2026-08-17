"""Add Razorpay SaaS billing tables.

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-17 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _uuid() -> postgresql.UUID:
    return postgresql.UUID(as_uuid=True)


def _jsonb() -> postgresql.JSONB:
    return postgresql.JSONB(astext_type=sa.Text())


def _now() -> sa.DateTime:
    return sa.DateTime(timezone=True)


def upgrade() -> None:
    op.execute(sa.text("CREATE SEQUENCE billing_invoice_number_seq START 1"))
    op.execute(sa.text("CREATE SEQUENCE billing_receipt_number_seq START 1"))

    op.create_table(
        "plans",
        sa.Column("id", _uuid(), nullable=False),
        sa.Column("parent_plan_id", _uuid(), nullable=True),
        sa.Column("razorpay_plan_id", sa.String(64), nullable=True),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("amount", sa.BigInteger(), nullable=False),
        sa.Column("currency", sa.String(3), nullable=False, server_default="INR"),
        sa.Column("interval", sa.String(16), nullable=False),
        sa.Column("interval_count", sa.BigInteger(), nullable=False, server_default="1"),
        sa.Column("trial_period_days", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("tax_rate", sa.Numeric(8, 6), nullable=False, server_default=sa.text("0.18")),
        sa.Column("refund_policy", sa.String(16), nullable=False, server_default="PRO_RATA"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("features", _jsonb(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("metadata_", _jsonb(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", _now(), nullable=False),
        sa.Column("updated_at", _now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", name="uq_plans_name"),
    )
    op.create_index("ix_plans_is_active", "plans", ["is_active"])

    op.create_table(
        "subscriptions",
        sa.Column("id", _uuid(), nullable=False),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column("plan_id", _uuid(), nullable=False),
        sa.Column("razorpay_subscription_id", sa.String(64), nullable=True),
        sa.Column("razorpay_customer_id", sa.String(64), nullable=True),
        sa.Column("status", sa.String(16), nullable=False, server_default="created"),
        sa.Column("current_period_start", _now(), nullable=True),
        sa.Column("current_period_end", _now(), nullable=True),
        sa.Column("trial_end", _now(), nullable=True),
        sa.Column("cancel_at_period_end", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("cancelled_at", _now(), nullable=True),
        sa.Column("ended_at", _now(), nullable=True),
        sa.Column("pause_start", _now(), nullable=True),
        sa.Column("pause_end", _now(), nullable=True),
        sa.Column("retry_count", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("max_retries", sa.BigInteger(), nullable=False, server_default="4"),
        sa.Column("currency_display", sa.String(3), nullable=False, server_default="INR"),
        sa.Column("trial_extension_count", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("deleted_at", _now(), nullable=True),
        sa.Column("version", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("metadata_", _jsonb(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", _now(), nullable=False),
        sa.Column("updated_at", _now(), nullable=False),
        sa.ForeignKeyConstraint(["plan_id"], ["plans.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "plan_id", name="uq_subscriptions_user_plan"),
    )
    op.create_index("ix_subscriptions_user_id", "subscriptions", ["user_id"])
    op.create_index("ix_subscriptions_razorpay_subscription_id", "subscriptions", ["razorpay_subscription_id"])
    op.create_index("ix_subscriptions_plan_id", "subscriptions", ["plan_id"])
    op.create_index("ix_subscriptions_id_version", "subscriptions", ["id", "version"])

    op.create_table(
        "payments",
        sa.Column("id", _uuid(), nullable=False),
        sa.Column("subscription_id", _uuid(), nullable=False),
        sa.Column("invoice_id", _uuid(), nullable=True),
        sa.Column("razorpay_payment_id", sa.String(64), nullable=False),
        sa.Column("razorpay_order_id", sa.String(64), nullable=True),
        sa.Column("amount", sa.BigInteger(), nullable=False),
        sa.Column("currency", sa.String(3), nullable=False, server_default="INR"),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("method", sa.String(16), nullable=True),
        sa.Column("captured_at", _now(), nullable=True),
        sa.Column("failed_at", _now(), nullable=True),
        sa.Column("error_code", sa.String(64), nullable=True),
        sa.Column("error_description", sa.Text(), nullable=True),
        sa.Column("refund_amount", sa.Numeric(20, 2), nullable=False, server_default=sa.text("0.0")),
        sa.Column("metadata_", _jsonb(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", _now(), nullable=False),
        sa.Column("updated_at", _now(), nullable=False),
        sa.ForeignKeyConstraint(["subscription_id"], ["subscriptions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("razorpay_payment_id", name="uq_payments_razorpay_payment_id"),
    )
    op.create_index("ix_payments_subscription_id", "payments", ["subscription_id"])
    op.create_index("ix_payments_invoice_id", "payments", ["invoice_id"])
    op.create_index("ix_payments_status", "payments", ["status"])

    op.create_table(
        "invoices",
        sa.Column("id", _uuid(), nullable=False),
        sa.Column("invoice_number", sa.String(32), nullable=False),
        sa.Column("subscription_id", _uuid(), nullable=False),
        sa.Column("payment_id", _uuid(), nullable=True),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="draft"),
        sa.Column("subtotal", sa.Numeric(20, 2), nullable=False),
        sa.Column("tax_rate", sa.Numeric(8, 6), nullable=False, server_default=sa.text("0.18")),
        sa.Column("tax_amount", sa.Numeric(20, 2), nullable=False),
        sa.Column("total", sa.Numeric(20, 2), nullable=False),
        sa.Column("currency", sa.String(3), nullable=False, server_default="INR"),
        sa.Column("seller_gstin", sa.String(15), nullable=False),
        sa.Column("buyer_gstin", sa.String(15), nullable=True),
        sa.Column("place_of_supply", sa.String(2), nullable=False),
        sa.Column("sac_code", sa.String(6), nullable=False, server_default="998314"),
        sa.Column("cgst_amount", sa.Numeric(20, 2), nullable=False, server_default=sa.text("0.0")),
        sa.Column("sgst_amount", sa.Numeric(20, 2), nullable=False, server_default=sa.text("0.0")),
        sa.Column("igst_amount", sa.Numeric(20, 2), nullable=False, server_default=sa.text("0.0")),
        sa.Column("issued_at", _now(), nullable=True),
        sa.Column("due_at", _now(), nullable=True),
        sa.Column("paid_at", _now(), nullable=True),
        sa.Column("pdf_url", sa.Text(), nullable=True),
        sa.Column("metadata_", _jsonb(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", _now(), nullable=False),
        sa.Column("updated_at", _now(), nullable=False),
        sa.ForeignKeyConstraint(["payment_id"], ["payments.id"]),
        sa.ForeignKeyConstraint(["subscription_id"], ["subscriptions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("invoice_number", name="uq_invoices_invoice_number"),
    )
    op.create_index("ix_invoices_subscription_id", "invoices", ["subscription_id"])
    op.create_index("ix_invoices_payment_id", "invoices", ["payment_id"])
    op.create_index("ix_invoices_user_id", "invoices", ["user_id"])
    op.create_index("ix_invoices_status", "invoices", ["status"])

    op.create_table(
        "invoice_line_items",
        sa.Column("id", _uuid(), nullable=False),
        sa.Column("invoice_id", _uuid(), nullable=False),
        sa.Column("plan_name", sa.String(128), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("quantity", sa.BigInteger(), nullable=False, server_default="1"),
        sa.Column("unit_price", sa.Numeric(20, 2), nullable=False),
        sa.Column("amount", sa.Numeric(20, 2), nullable=False),
        sa.Column("tax_amount", sa.Numeric(20, 2), nullable=False, server_default=sa.text("0.0")),
        sa.Column("sac_code", sa.String(6), nullable=False, server_default="998314"),
        sa.ForeignKeyConstraint(["invoice_id"], ["invoices.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_invoice_line_items_invoice_id", "invoice_line_items", ["invoice_id"])

    op.create_table(
        "payment_receipts",
        sa.Column("id", _uuid(), nullable=False),
        sa.Column("receipt_number", sa.String(32), nullable=False),
        sa.Column("subscription_id", _uuid(), nullable=False),
        sa.Column("payment_id", _uuid(), nullable=False),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column("amount", sa.Numeric(20, 2), nullable=False),
        sa.Column("currency", sa.String(3), nullable=False, server_default="INR"),
        sa.Column("payment_method", sa.String(16), nullable=True),
        sa.Column("razorpay_payment_id", sa.String(64), nullable=False),
        sa.Column("receipt_date", _now(), nullable=False),
        sa.Column("billing_period_start", _now(), nullable=True),
        sa.Column("billing_period_end", _now(), nullable=True),
        sa.Column("plan_name", sa.String(128), nullable=True),
        sa.Column("pdf_url", sa.Text(), nullable=True),
        sa.Column("created_at", _now(), nullable=False),
        sa.ForeignKeyConstraint(["payment_id"], ["payments.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["subscription_id"], ["subscriptions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("receipt_number", name="uq_payment_receipts_receipt_number"),
        sa.UniqueConstraint("payment_id", name="uq_payment_receipts_payment_id"),
    )
    op.create_index("ix_payment_receipts_subscription_id", "payment_receipts", ["subscription_id"])
    op.create_index("ix_payment_receipts_user_id", "payment_receipts", ["user_id"])

    op.create_table(
        "webhook_events",
        sa.Column("id", _uuid(), nullable=False),
        sa.Column("razorpay_event_id", sa.String(64), nullable=False),
        sa.Column("event_type", sa.String(64), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("payload", _jsonb(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("processed_at", _now(), nullable=True),
        sa.Column("failed_at", _now(), nullable=True),
        sa.Column("error_message", sa.String(1000), nullable=True),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", _now(), nullable=False),
        sa.Column("updated_at", _now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("razorpay_event_id", name="uq_webhook_events_razorpay_event_id"),
    )
    op.create_index("ix_webhook_events_event_type", "webhook_events", ["event_type"])
    op.create_index("ix_webhook_events_status", "webhook_events", ["status"])

    op.create_table(
        "audit_logs",
        sa.Column("id", _uuid(), nullable=False),
        sa.Column("entity_type", sa.String(32), nullable=False),
        sa.Column("entity_id", sa.String(64), nullable=False),
        sa.Column("action", sa.String(64), nullable=False),
        sa.Column("user_id", sa.String(255), nullable=True),
        sa.Column("ip_address", sa.String(64), nullable=True),
        sa.Column("user_agent", sa.String(512), nullable=True),
        sa.Column("changes", _jsonb(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("metadata_", _jsonb(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", _now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_audit_logs_entity_type_id", "audit_logs", ["entity_type", "entity_id"])
    op.create_index("ix_audit_logs_action", "audit_logs", ["action"])
    op.create_index("ix_audit_logs_created_at", "audit_logs", ["created_at"])

    op.create_table(
        "invoice_voids",
        sa.Column("id", _uuid(), nullable=False),
        sa.Column("original_invoice_id", _uuid(), nullable=False),
        sa.Column("void_reason", sa.String(32), nullable=False),
        sa.Column("void_description", sa.String(500), nullable=True),
        sa.Column("voided_by_user_id", sa.String(255), nullable=False),
        sa.Column("voided_at", _now(), nullable=False),
        sa.Column("original_invoice_number", sa.String(32), nullable=False),
        sa.Column("original_subtotal", sa.Numeric(20, 2), nullable=False),
        sa.Column("original_tax_rate", sa.Numeric(8, 6), nullable=False),
        sa.Column("original_tax_amount", sa.Numeric(20, 2), nullable=False),
        sa.Column("original_total", sa.Numeric(20, 2), nullable=False),
        sa.Column("original_currency", sa.String(3), nullable=False, server_default="INR"),
        sa.Column("reissued_invoice_id", _uuid(), nullable=True),
        sa.Column("created_at", _now(), nullable=False),
        sa.ForeignKeyConstraint(["original_invoice_id"], ["invoices.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["reissued_invoice_id"], ["invoices.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("original_invoice_id", name="uq_invoice_voids_original_invoice"),
    )

    op.create_table(
        "trial_extensions",
        sa.Column("id", _uuid(), nullable=False),
        sa.Column("subscription_id", _uuid(), nullable=False),
        sa.Column("requested_days", sa.Integer(), nullable=False),
        sa.Column("approved_days", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("requested_by_user_id", sa.String(255), nullable=False),
        sa.Column("requested_at", _now(), nullable=False),
        sa.Column("approved_by_user_id", sa.String(255), nullable=True),
        sa.Column("approved_at", _now(), nullable=True),
        sa.Column("rejection_reason", sa.String(500), nullable=True),
        sa.Column("original_trial_end", _now(), nullable=True),
        sa.Column("new_trial_end", _now(), nullable=True),
        sa.Column("created_at", _now(), nullable=False),
        sa.Column("updated_at", _now(), nullable=False),
        sa.ForeignKeyConstraint(["subscription_id"], ["subscriptions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_trial_extensions_subscription_id", "trial_extensions", ["subscription_id"])

    op.create_table(
        "email_templates",
        sa.Column("id", _uuid(), nullable=False),
        sa.Column("email_type", sa.String(32), nullable=False),
        sa.Column("subject", sa.String(200), nullable=False),
        sa.Column("body_html", sa.Text(), nullable=False),
        sa.Column("body_plain", sa.Text(), nullable=False),
        sa.Column("variables", _jsonb(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", _now(), nullable=False),
        sa.Column("updated_at", _now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email_type", name="uq_email_templates_email_type"),
    )

    op.create_table(
        "invoice_batches",
        sa.Column("id", _uuid(), nullable=False),
        sa.Column("batch_name", sa.String(200), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("plan_ids", _jsonb(), nullable=True),
        sa.Column("status_filter", _jsonb(), nullable=True),
        sa.Column("date_from", _now(), nullable=True),
        sa.Column("date_to", _now(), nullable=True),
        sa.Column("total_subscriptions", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("invoices_generated", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("failed_count", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("failed_subscriptions", _jsonb(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("initiated_by_user_id", sa.String(255), nullable=False),
        sa.Column("initiated_at", _now(), nullable=False),
        sa.Column("completed_at", _now(), nullable=True),
        sa.Column("pdf_urls", _jsonb(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("created_at", _now(), nullable=False),
        sa.Column("updated_at", _now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_invoice_batches_status", "invoice_batches", ["status"])

    op.create_table(
        "currencies",
        sa.Column("id", _uuid(), nullable=False),
        sa.Column("code", sa.String(3), nullable=False),
        sa.Column("name", sa.String(64), nullable=False),
        sa.Column("symbol", sa.String(8), nullable=False),
        sa.Column("iso_number", sa.BigInteger(), nullable=False),
        sa.Column("decimal_places", sa.BigInteger(), nullable=False, server_default="2"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", _now(), nullable=False),
        sa.Column("updated_at", _now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("code", name="uq_currencies_code"),
    )

    op.create_table(
        "fx_rates",
        sa.Column("id", _uuid(), nullable=False),
        sa.Column("base_currency", sa.String(3), nullable=False),
        sa.Column("target_currency", sa.String(3), nullable=False),
        sa.Column("rate", sa.Numeric(20, 6), nullable=False),
        sa.Column("source", sa.String(16), nullable=False, server_default="razorpay"),
        sa.Column("effective_at", _now(), nullable=False),
        sa.Column("expires_at", _now(), nullable=True),
        sa.Column("fetched_at", _now(), nullable=True),
        sa.Column("manually_entered_by_user_id", sa.String(255), nullable=True),
        sa.Column("created_at", _now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "base_currency", "target_currency", "effective_at", name="uq_fx_rates_pair_period"
        ),
    )
    op.create_index("ix_fx_rates_pair", "fx_rates", ["base_currency", "target_currency"])

    op.create_table(
        "reports",
        sa.Column("id", _uuid(), nullable=False),
        sa.Column("report_type", sa.String(32), nullable=False),
        sa.Column("report_name", sa.String(200), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("date_from", _now(), nullable=True),
        sa.Column("date_to", _now(), nullable=True),
        sa.Column("plan_ids", _jsonb(), nullable=True),
        sa.Column("user_ids", _jsonb(), nullable=True),
        sa.Column("generated_at", _now(), nullable=True),
        sa.Column("generated_by_user_id", sa.String(255), nullable=True),
        sa.Column("output_format", sa.String(8), nullable=False, server_default="csv"),
        sa.Column("file_url", sa.String(1000), nullable=True),
        sa.Column("row_count", sa.BigInteger(), nullable=True),
        sa.Column("created_at", _now(), nullable=False),
        sa.Column("updated_at", _now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_reports_report_type", "reports", ["report_type"])
    op.create_index("ix_reports_status", "reports", ["status"])


def downgrade() -> None:
    for table in (
        "reports",
        "fx_rates",
        "currencies",
        "invoice_batches",
        "email_templates",
        "trial_extensions",
        "invoice_voids",
        "audit_logs",
        "webhook_events",
        "payment_receipts",
        "invoice_line_items",
        "invoices",
        "payments",
        "subscriptions",
        "plans",
    ):
        op.drop_table(table)
    op.execute(sa.text("DROP SEQUENCE IF EXISTS billing_receipt_number_seq"))
    op.execute(sa.text("DROP SEQUENCE IF EXISTS billing_invoice_number_seq"))
