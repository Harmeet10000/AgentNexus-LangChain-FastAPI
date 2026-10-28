"""Stable task wire names shared by producers and consumers."""

from typing import Final

DOCUMENTS_INGEST: Final = "tasks.documents_ingest"
DOCUMENTS_INGEST_TERMINAL: Final = "tasks.documents_ingest_terminal"
CRAWLER_CRAWL: Final = "tasks.crawler_crawl"
INGESTION_TASK_NAMES: Final[frozenset[str]] = frozenset(
    {DOCUMENTS_INGEST, DOCUMENTS_INGEST_TERMINAL}
)
SEND_VERIFICATION_EMAIL: Final = "auth.send_verification_email"
SEND_PASSWORD_RESET_EMAIL: Final = "auth.send_password_reset_email"  # noqa: S105 — wire name, not a credential
AUTH_PENDING_REVOCATION_RECONCILE: Final = "auth.pending_revocation_reconcile"
BILLING_RENEWAL: Final = "billing.renewal"
BILLING_DUNNING: Final = "billing.dunning"
BILLING_INVOICE_GENERATION: Final = "billing.invoice_generation"
BILLING_RECEIPT_GENERATION: Final = "billing.receipt_generation"
BILLING_PAUSE_RESUME: Final = "billing.pause_resume"
BILLING_RECONCILIATION: Final = "billing.reconciliation"
BILLING_PLAN_RECOVERY: Final = "billing.plan_recovery"
CREDITS_EXPIRE: Final = "credits.expire"
CREDITS_RECONCILE: Final = "credits.reconcile"
AGENT_MEMORY_CONSOLIDATION: Final = "tasks.agent_memory_consolidation"
AGENT_SAUL_TRANSCRIPT_SWEEP: Final = "tasks.agent_saul_transcript_sweep"
