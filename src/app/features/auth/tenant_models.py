"""SQL tenant ownership and tenant-scoped legal membership models."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, ForeignKey, Index, Integer, String
from sqlalchemy.orm import (  # noqa: TC002 — SQLAlchemy resolves annotations at runtime
    Mapped,
    mapped_column,
)

from database.base import Base


class Tenant(Base):
    # Cognee owns the generic public ``tenants`` table in the shared TSDB.
    # Keep application authorization data in an explicitly owned namespace so
    # Alembic can never reinterpret or mutate Cognee's schema.
    __tablename__ = "app_tenants"

    id: Mapped[str] = mapped_column(String(length=255), primary_key=True)
    display_name: Mapped[str] = mapped_column(String(length=255), nullable=False)
    self_review_allowed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        nullable=False, default=lambda: datetime.now(tz=UTC)
    )


class TenantMembership(Base):
    __tablename__ = "app_tenant_memberships"
    __table_args__ = (Index("ix_app_tenant_memberships_user_id", "user_id"),)

    tenant_id: Mapped[str] = mapped_column(
        String(length=255), ForeignKey("app_tenants.id", ondelete="CASCADE"), primary_key=True
    )
    user_id: Mapped[str] = mapped_column(String(length=255), primary_key=True)
    legal_role: Mapped[str] = mapped_column(String(length=32), nullable=False, default="analyst")
    # ``legal_role`` is retained as the wire-compatible primary role while the
    # JSON list is the authoritative representation for future multi-role
    # membership.  Mongo ``User.legal_roles`` is a mirror for auth reads.
    legal_roles: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=lambda: ["analyst"]
    )
    role_revision: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # SQL remains authoritative while Mongo is a denormalised mirror.  A
    # failed mirror write is recorded here for a resumable reconciler instead
    # of being reported as a failed membership mutation after SQL committed.
    mirror_sync_pending: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    mirror_sync_error: Mapped[str | None] = mapped_column(String(length=500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        nullable=False, default=lambda: datetime.now(tz=UTC)
    )
