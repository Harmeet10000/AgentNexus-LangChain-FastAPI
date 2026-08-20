"""SQLAlchemy base model for all database tables."""

from __future__ import annotations

from datetime import datetime
from re import compile as re_compile
from typing import override

from sqlalchemy import DateTime, MetaData, String, func
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.inspection import inspect
from sqlalchemy.orm import (  # noqa: TC002 — Mapped resolved at runtime by SQLAlchemy mapper
    DeclarativeBase,
    Mapped,
    declared_attr,
    mapped_column,
)

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

_CAMEL_TO_SNAKE = re_compile(r"(?<!^)(?=[A-Z])")


class Base(AsyncAttrs, DeclarativeBase):
    """Base class for all SQLAlchemy models.

    Features:
    - Timezone-aware datetime columns via type_annotation_map
    - Auto-generated __tablename__ from class name (camelCase → snake_case)
    - Optional timestamp mixin for audit trails
    - Custom __repr__ for debug-friendly log output
    """

    metadata = MetaData(naming_convention=NAMING_CONVENTION)

    type_annotation_map = {  # noqa: RUF012 — SQLAlchemy reads this class dict
        datetime: DateTime(timezone=True),
    }

    @override
    def __repr__(self) -> str:
        """Return a debug-friendly representation using only column attributes.

        Scoping to column_attrs avoids triggering lazy-loaded relationship
        attributes, which would cause MissingGreenlet errors when logging
        objects outside of an async context.
        """
        mapper = inspect(self)
        cols = ", ".join(
            f"{c.key}={getattr(self, c.key)!r}"
            for c in mapper.mapper.column_attrs
            if hasattr(self, c.key)
        )
        return f"{type(self).__name__}({cols})"

    @declared_attr.directive
    @override
    def __tablename__(cls) -> str:  # noqa: N805 — SQLAlchemy declared_attr passes the class
        """Auto-generate table name from class name: User → user, ChatSession → chat_session."""
        return _CAMEL_TO_SNAKE.sub("_", cls.__name__).lower()


class TimestampMixin:
    """Opt-in mixin for audit timestamps.

    Use server_default=func.now() to ensure a single source of truth for
    timestamps (the database clock) rather than Python-side defaults that
    can drift across horizontally scaled app instances.
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class PublicIdMixin:
    """Opt-in mixin for opaque public identifiers.

    Use this when a table's primary key crosses the API boundary.
    Prefixes (doc_, session_, etc.) make entity types visually catchable
    in logs and bug reports.
    """

    public_id: Mapped[str] = mapped_column(
        String(32),
        unique=True,
        index=True,
        nullable=False,
    )
