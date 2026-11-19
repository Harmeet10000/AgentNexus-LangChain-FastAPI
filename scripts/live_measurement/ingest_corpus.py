"""Ingest the golden corpus artifacts through the production upload command.

Runs in-process (no HTTP rate limiter, no broker round-trip for the upload
itself — the command still enqueues the real Celery ingestion task on
CloudAMQP, which the local worker processes). Idempotent: content-hash
duplicates return the existing document id, so reruns resume cleanly.

Greptile wave-3 fixes (both findings on the original driver):
- ONE COMMITTED TRANSACTION PER ARTIFACT. ``upload_document()`` only flushes
  the document row and stages the outbox event — the normal HTTP dependency
  supplies the commit, which no script dependency does. The old driver saved
  the mapping while both rows were still uncommitted: workers could not see
  the upload and a later rollback left mapped ids with no document behind
  them. Each artifact now commits BEFORE its mapping entry is written.
- STATUS-AWARE RESUME. Every mapped source used to be skipped, including
  documents whose ingestion later failed, so a rerun after a quota reset
  could not retry them (and deleting the mapping entry would not help:
  ``upload_document()`` returns the existing document without queuing new
  work). Mapped sources now skip only on a SUCCESSFUL terminal status; a
  failed or still-parked document is reported for the explicit re-drive path
  (``reingest_failed_inprocess.py``) instead of being silently skipped.

Reads .testing-credentials.json for the principal.
Writes .live-ingestion-mapping.json: {source_id: {doc_id, status, filename}}.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from returns.result import Failure
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.connections.postgres import get_database_url

REPO = Path(__file__).resolve().parents[2]
CREDS = REPO / ".testing-credentials.json"
MAPPING = REPO / ".live-ingestion-mapping.json"
ARTIFACTS = REPO / "evals" / "corpus" / "artifacts"
# SRC-05/SRC-06 originals are intentionally not in-repo (manifest note) and
# referenced by zero golden rows.
SKIP_SOURCES = {"SRC-05", "SRC-06"}

# Only these statuses mean "the corpus artifact is searchable; skip it".
_SUCCESSFUL_TERMINAL_STATUSES = {"completed", "completed_with_warnings"}

_CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".md": "text/markdown",
    ".html": "text/html",
}


async def _build_session_factory():
    from app.connections.postgres import init_db

    _, session_factory = await init_db()
    return session_factory


async def _document_status(engine, doc_id: str) -> str | None:
    """The document's CURRENT persisted status, or None when the row is gone."""
    async with engine.connect() as conn:
        row = await conn.execute(
            text("select status from documents where id = :id"), {"id": doc_id}
        )
        found = row.first()
        return str(found[0]) if found is not None else None


async def main() -> None:
    from app.config import get_settings
    from app.features.documents.adapters.persistence.postgres.document_repository import (
        DocumentRepository,
    )
    from app.features.documents.application.commands.upload_document import (
        DocumentCommandService,
    )
    from app.utils.storage import StorageService

    creds = json.loads(CREDS.read_text())
    settings = get_settings()
    mapping: dict[str, dict[str, str]] = json.loads(MAPPING.read_text()) if MAPPING.exists() else {}

    session_factory = await _build_session_factory()
    engine = create_async_engine(get_database_url(), pool_pre_ping=True)
    object_store = StorageService.from_settings(settings=settings)
    # The command service owns this session for the whole run; the script
    # supplies the commit the HTTP dependency normally supplies.
    session = session_factory()
    service = DocumentCommandService(DocumentRepository(session), object_store)

    artifacts = sorted(
        p for p in ARTIFACTS.glob("*/original.*") if p.parent.name not in SKIP_SOURCES
    )
    print(f"ingesting {len(artifacts)} artifacts as {creds['email']}")
    for artifact in artifacts:
        source_id = artifact.parent.name
        if source_id in mapping and mapping[source_id].get("doc_id"):
            status = await _document_status(engine, mapping[source_id]["doc_id"])
            if status in _SUCCESSFUL_TERMINAL_STATUSES:
                print(f"skip {source_id}: already ingested ({status})")
                continue
            # Not a successful terminal status: re-uploading cannot help
            # (content-hash duplicate returns the existing document without
            # queuing new work) — the explicit re-drive path owns it.
            print(
                f"retry-needed {source_id}: doc {mapping[source_id]['doc_id']} status={status!r}; "
                "re-drive with scripts/live_measurement/reingest_failed_inprocess.py"
            )
            mapping[source_id]["status"] = status or "missing"
            continue
        raw = artifact.read_bytes()
        result = await service.upload_document(
            user_id=creds["user_id"],
            tenant_id=creds.get("tenant_id"),
            filename=f"{source_id}{artifact.suffix}",
            content_type=_CONTENT_TYPES.get(artifact.suffix, "application/octet-stream"),
            raw_bytes=raw,
        )
        if isinstance(result, Failure):
            failure = result.failure()
            print(f"FAIL {source_id}: {getattr(failure, 'message', failure)}")
            continue
        payload = result.unwrap()
        # Commit THIS artifact's transaction before recording it: an uncommitted
        # mapping entry points at a document no worker can see.
        await session.commit()
        mapping[source_id] = {
            "doc_id": payload.doc_id,
            "status": payload.status,
            "filename": artifact.name,
            "duplicate": getattr(payload, "duplicate", False),
        }
        MAPPING.write_text(json.dumps(mapping, indent=2))
        print(f"queued+committed {source_id} -> {payload.doc_id} ({payload.status})")

    await session.close()
    await engine.dispose()
    print(f"mapping written: {MAPPING}")


if __name__ == "__main__":
    asyncio.run(main())
