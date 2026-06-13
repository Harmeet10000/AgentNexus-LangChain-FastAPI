"""Startup readiness checks for the unified document pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import text

from app.utils import ServiceUnavailableException

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from app.shared.services.storage import StorageService


async def run_document_startup_checks(*, engine: AsyncEngine, object_store: StorageService) -> None:
    await object_store.verify_access()
    async with engine.connect() as connection:
        extensions = await connection.execute(
            text(
                """
                SELECT extname
                FROM pg_extension
                WHERE extname IN ('vector', 'pg_textsearch', 'pg_trgm')
                """
            )
        )
        installed = {str(row[0]) for row in extensions.fetchall()}
        missing = {"vector", "pg_textsearch", "pg_trgm"} - installed
        if missing:
            raise ServiceUnavailableException(
                detail="Document search extensions are missing",
                data={"missing_extensions": sorted(missing)},
            )

        indexes = await connection.execute(
            text(
                """
                SELECT to_regclass('chunks_bm25_idx'),
                       to_regclass('chunks_embedding_idx'),
                       to_regclass('chunks_search_text_trgm_idx')
                """
            )
        )
        row = indexes.fetchone()
        if row is None or any(value is None for value in row):
            raise ServiceUnavailableException(
                detail="Document search indexes are missing",
                data={
                    "required_indexes": [
                        "chunks_bm25_idx",
                        "chunks_embedding_idx",
                        "chunks_search_text_trgm_idx",
                    ]
                },
            )
