"""Branch statements prepare and execute with default bindings (retrieval-sql 3.4).

The deleted monolithic query could not be prepared by either driver once every
optional binding sat at its default (`:x IS NULL` dual-context parameters fail
prepare for None *and* concrete bindings). The branch statements replaced it
with CAST-typed parameters; this test pins that property by running every
branch end-to-end with defaults. It also pins the production invariant the
bare keyword-index name relies on: exactly one matching index per retrieval
index name in the deployed schema.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text as sa_text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.connections.postgres import get_database_url
from app.features.documents.repository import (
    DocumentRepository,
    build_search_filter_params,
)

pytestmark = [pytest.mark.integration, pytest.mark.requires_db]

USER_ID = "scratch-user-00"
QUERY = "termination obligations compensation"

RETRIEVAL_INDEXES = (
    "chunks_bm25_idx",
    "chunks_embedding_idx",
    "chunks_search_text_trgm_idx",
)


async def _session_factory() -> object:
    engine = create_async_engine(get_database_url(flavour="async"))
    return async_sessionmaker(engine, expire_on_commit=False)


async def test_branch_methods_execute_with_default_bindings() -> None:
    factory = await _session_factory()
    async with factory() as session:
        await session.execute(sa_text("SET search_path TO scratch_13, public"))
        repo = DocumentRepository(session)
        embedding_row = await session.execute(
            sa_text(
                "SELECT embedding::text FROM chunks WHERE user_id = :user "
                "AND embedding IS NOT NULL LIMIT 1"
            ),
            {"user": USER_ID},
        )
        embedding = [float(value) for value in embedding_row.scalar_one()[1:-1].split(",")]
        filter_params = build_search_filter_params(metadata_filter={})

        bm25 = await repo.bm25_search(
            user_id=USER_ID,
            query=QUERY,
            candidate_limit=50,
            filter_params=dict(filter_params),
        )
        vector = await repo.vector_search(
            user_id=USER_ID,
            embedding=embedding,
            candidate_limit=50,
            filter_params=dict(filter_params),
        )
        trigram = await repo.trigram_search(
            user_id=USER_ID,
            query=QUERY,
            candidate_limit=50,
            filter_params=dict(filter_params),
        )
        from returns.result import Success

        assert isinstance(bm25, Success), bm25
        assert isinstance(vector, Success), vector
        assert isinstance(trigram, Success), trigram
        await session.rollback()


async def test_deployed_schema_has_unique_retrieval_index_names() -> None:
    factory = await _session_factory()
    async with factory() as session:
        for index in RETRIEVAL_INDEXES:
            count = await session.execute(
                sa_text(
                    "SELECT count(*) FROM pg_indexes "
                    "WHERE schemaname = 'public' AND indexname = :name"
                ),
                {"name": index},
            )
            assert count.scalar_one() == 1, f"duplicate or missing index: {index}"
        await session.rollback()
