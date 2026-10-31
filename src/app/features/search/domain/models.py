"""Site-search domain vocabulary.

Pure domain types only: no SQLAlchemy, no FastAPI, no I/O. The persistence
shapes for this feature live in ``adapters/persistence/models.py`` because the
task brief places the ORM there (documents keeps its aggregates in ``domain/``;
search follows the brief, not that precedent).
"""

from __future__ import annotations

from enum import StrEnum


class SearchDocumentStatus(StrEnum):
    """Lifecycle of a search document's chunk/embed pipeline."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

    @classmethod
    def coerce(cls, raw: object) -> SearchDocumentStatus:
        """Map any stored value onto the enum, defaulting unknowns to PENDING.

        The column is a plain varchar so rolling deploys and legacy rows never
        crash a read; an unrecognized value degrades to the pre-pipeline state.
        """
        try:
            return cls(str(raw))
        except ValueError:
            return cls.PENDING
