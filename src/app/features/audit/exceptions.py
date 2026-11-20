"""Compatibility imports for pre-Phase-1 audit error consumers."""

from .domain.errors import (
    AuditCode,
    AuditError,
    AuditInfrastructureError,
    AuditResult,
)

__all__ = [
    "AuditCode",
    "AuditError",
    "AuditInfrastructureError",
    "AuditResult",
]
