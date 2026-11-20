"""Compatibility import for untouched consumers during the slice migration."""

from .adapters.outbound.postgres.repository import (
    AuditLogRepository,
    PostgresAuditLogRepository,
)

__all__ = ["AuditLogRepository", "PostgresAuditLogRepository"]
