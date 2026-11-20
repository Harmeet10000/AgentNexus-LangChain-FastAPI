"""Mappings local to the PostgreSQL audit adapter."""

from app.features.audit.domain.entities.audit_entry import AuditEntry

from .models import AuditLog


def to_model(entry: AuditEntry) -> AuditLog:
    return AuditLog(
        id=entry.id,
        entity_type=entry.entity_type,
        entity_id=entry.entity_id,
        action=entry.action,
        user_id=entry.user_id,
        ip_address=entry.ip_address,
        user_agent=entry.user_agent,
        changes=dict(entry.changes),
        metadata_=dict(entry.metadata_),
        created_at=entry.created_at,
    )


def to_entity(row: AuditLog) -> AuditEntry:
    return AuditEntry(
        id=row.id,
        entity_type=row.entity_type,
        entity_id=row.entity_id,
        action=row.action,
        user_id=row.user_id,
        ip_address=row.ip_address,
        user_agent=row.user_agent,
        changes=dict(row.changes),
        metadata_=dict(row.metadata_),
        created_at=row.created_at,
    )
