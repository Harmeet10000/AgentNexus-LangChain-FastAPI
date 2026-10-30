"""SQLAlchemy expression binds for Postgres-specific value shapes (asyncpg-safe).

The repeatable pattern for writing repository queries as SQLAlchemy core
expressions instead of raw ``text()`` SQL:

1. **Query against the ORM models** (``UnifiedDocument``, ``UnifiedChunk``,
   ``Invoice``, ``ChatSession``) — never against hand-named table strings.
2. **Membership against uuid[]** uses ``== any_()`` over a typed ARRAY bind
   (:func:`uuid_array_param`), which renders ``col = ANY(CAST(:x AS uuid[]))``
   — the exact shape the driver requires. Plain ``col.in_([str, ...])`` binds
   each element through the column's uuid type and is acceptable, but the
   ANY-cast form is the repo standard because it keeps the single-parameter
   binding surface the asyncpg driver and the bind-cast gate tests rely on.
3. **A string uuid key** uses :func:`uuid_param`, which renders
   ``CAST(:x AS uuid)``. Never write ``:x::uuid`` — the asyncpg dialect cannot
   parse a cast attached directly to a bind placeholder.
4. **NULL-guarded optional filters** (``:param IS NULL OR ...``) become
   Python-side conditionals on the already-known parameter value, or
   ``bindparam(...).is_(None)`` when the parameter's presence is itself
   dynamic inside one statement.
5. **Fallback**: keep ``text()`` with explicit bindparams for queries built on
   Postgres extension operators that have no readable expression form (here:
   pg_textsearch ``<@>``, DiskANN ``<=>``, pg_trgm ``%``/``similarity``) and
   for session/maintenance commands (``SET LOCAL``, ``ANALYZE``).

Why the casts exist: the application pool runs the asyncpg driver, which
prepares statements and type-checks every bind; an untyped string reaching a
uuid parameter is rejected at encode time. The explicit CAST keeps the bind a
text value end-to-end and lets Postgres do the conversion, exactly like the
``text()`` statements this module replaced.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import String, any_, bindparam, cast
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.dialects.postgresql import UUID as PGUUID

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from sqlalchemy import ColumnElement

# Subscripted at the type level so the generic lands as TypeEngine[object]
# for `ty`; at runtime this is the plain non-native-uuid PGUUID type.
_PGUUID_TEXT: Any = PGUUID(as_uuid=False)
_PGUUID_ARRAY: Any = ARRAY(PGUUID(as_uuid=False))


def uuid_param(name: str, value: str) -> ColumnElement[Any]:
    """A string uuid bind rendered as ``CAST(:name AS uuid)`` (asyncpg-safe)."""
    return cast(bindparam(name, value=value, type_=String), _PGUUID_TEXT)


def uuid_array_param(name: str, values: Sequence[str]) -> ColumnElement[Any]:
    """A uuid[] bind rendered as ``CAST(:name AS uuid[])`` (asyncpg-safe)."""
    return cast(
        bindparam(name, value=list(values), type_=ARRAY(String())),
        _PGUUID_ARRAY,
    )


def uuid_any(name: str, values: Sequence[str]) -> ColumnElement[bool]:
    """``col == any_(uuid_array_param(name, values))`` — the ``= ANY(uuid[])`` filter."""
    return any_(uuid_array_param(name=name, values=values))
