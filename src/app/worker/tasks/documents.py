"""Compatibility shim for the canonical document worker adapter.

Document pipeline internals remain outside this refactor.
"""

from app.worker.jobs.documents import ingest_document

__all__ = ["ingest_document"]
