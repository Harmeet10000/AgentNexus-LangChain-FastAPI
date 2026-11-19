"""Re-ingest failed documents IN-PROCESS through the production ingestion graph.

Assembles the same runtime the Celery worker builds (engine, sessions, redis,
graphiti, object store, idempotency, extraction, graph writer) and calls the
production ``run_document_ingestion_task`` per failed document — the worker
path minus the broker transport. Sequential to respect provider rate limits.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


async def main() -> None:
    from sqlalchemy import text

    from app.config import get_settings
    from app.utils.storage import StorageService
    from sqlalchemy.ext.asyncio import create_async_engine

    from app.connections.postgres import get_database_url, init_db
    from app.extraction.langextract.service import AsyncExtractionService
    from app.features.documents.application.commands.upload_document import (
        process_document_ingestion,
        run_document_ingestion_task,
    )
    from app.features.documents.adapters.persistence.postgres.document_repository import (
        DocumentRepository,
    )
    from app.resilience.idempotency import IdempotencyGuard
    from app.lifecycle.graphs import graphiti_service
    from app.connections.graphiti import setup_graphiti, setup_graphiti_indices
    settings = get_settings()
    creds = json.loads((REPO / ".testing-credentials.json").read_text())
    mapping = json.loads((REPO / ".live-ingestion-mapping.json").read_text())

    engine, session_local = await init_db()
    redis_url = settings.REDIS_URL
    from app.connections.redis import create_redis_client

    redis = create_redis_client(redis_url)
    await redis.ping()
    graphiti = None
    for attempt in range(1, 6):
        try:
            graphiti = await setup_graphiti(
                neo4j_uri=settings.NEO4J_URI,
                neo4j_user=settings.NEO4J_USERNAME,
                neo4j_password=settings.NEO4J_PASSWORD.get_secret_value(),
                neo4j_database=settings.NEO4J_DATABASE,
            )
            await setup_graphiti_indices(graphiti)
            print(f"graphiti ready (attempt {attempt})", flush=True)
            break
        except Exception as exc:  # noqa: BLE001 — Aura free tier wakes slowly
            print(f"graphiti attempt {attempt} failed: {exc!r}", flush=True)
            if attempt == 5:
                print("graphiti unavailable after 5 attempts — Neo4j Aura unreachable; aborting run", flush=True)
                raise
            await asyncio.sleep(30)
    object_store = StorageService.from_settings(settings=settings)
    idempotency = IdempotencyGuard(redis=redis, db_engine=engine, require_durable=True)

    from app.lifecycle.graphs import provide_document_ingestion_graph

    graph = provide_document_ingestion_graph(
        settings=settings,
        object_store=object_store,
        graphiti=graphiti,
        ingest_document_fn=process_document_ingestion,
        extraction=AsyncExtractionService.from_settings(settings),
        graph_writer=graphiti_service(graphiti),
        idempotency=idempotency,
    )

    engine2 = create_async_engine(get_database_url(), pool_pre_ping=True)
    failed: list[dict[str, str]] = []
    async with engine2.connect() as conn:
        for sid, m in mapping.items():
            res = await conn.execute(
                text("select object_uri, metadata_->>'content_type' as ct, status from documents where id = :id"),
                {"id": m["doc_id"]},
            )
            row = res.first()
            # Re-run everything except a SUCCESSFUL terminal status. Quota-failed
            # documents park as quota_exhausted; exhausted worker attempts save
            # a terminal `failed` — and `run_document_ingestion_task` restarts
            # those by setting `processing` again. Excluding `failed` (the old
            # driver did) meant running this script after the quota reset
            # silently skipped exactly the documents it exists to recover
            # (Greptile wave-3 finding). Content-hash/idempotency dedupe keeps
            # the re-drive safe.
            if row is not None and row[2] not in ("completed", "completed_with_warnings"):
                failed.append(
                    {
                        "source_id": sid,
                        "doc_id": m["doc_id"],
                        "object_uri": row[0],
                        "content_type": row[1] or "application/octet-stream",
                    }
                )
    await engine2.dispose()
    print(f"non-terminal documents to re-ingest: {len(failed)}", flush=True)

    for entry in failed:
        print(f"--- re-ingesting {entry['source_id']} ({entry['doc_id']})", flush=True)
        try:
            result = await run_document_ingestion_task(
                document_id=entry["doc_id"],
                user_id=creds["user_id"],
                tenant_id=creds.get("tenant_id"),
                filename=f"{entry['source_id']}.pdf",
                content_type=entry["content_type"],
                object_uri=entry["object_uri"],
                document_version=1,
                graph=graph,
                engine=engine,
                session_local=session_local,
                idempotency=idempotency,
            )
            print(f"    result: {json.dumps(result)[:220]}", flush=True)
        except Exception as exc:  # noqa: BLE001 — the driver records every failure
            print(f"    EXCEPTION: {exc!r}", flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
