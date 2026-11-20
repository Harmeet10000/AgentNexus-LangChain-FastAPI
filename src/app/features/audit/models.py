"""Compatibility import for the pre-Phase-1 model registry."""

from .adapters.outbound.postgres.models import AuditLog
from .domain.enums import AuditAction

__all__ = ["AuditAction", "AuditLog"]
