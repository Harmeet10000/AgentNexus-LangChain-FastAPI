"""Compatibility imports for pre-Phase-1 audit error consumers."""

from .domain.errors import (
    AuditCode,
    AuditError,
    AuditInfrastructureError,
    AuditResult,
    audit_error_to_http_status,
)

__all__ = [
    "AuditCode",
    "AuditError",
    "AuditInfrastructureError",
    "AuditResult",
    "audit_error_to_http_status",
]
