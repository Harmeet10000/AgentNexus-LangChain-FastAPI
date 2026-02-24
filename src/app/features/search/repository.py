from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.features.search.model import DocumentVector


class SearchRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def hybrid_search(
        self,
        query: str,
        embedding: list[float],
        user_id: str,
        limit: int,
        offset: int,
        candidate_limit: int,
        rrf_k: int,
    ) -> tuple[list[dict[str, Any]], bool]:
        """
        Reciprocal Rank Fusion over vector search (pgvectorscale / cosine distance)
        and keyword search (pg_textsearch / BM25).

        Returns (results, has_more).
        """
        # Convert embedding list to PostgreSQL vector format string
        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

        sql = """
        WITH vector_search AS (
            SELECT id,
                   ROW_NUMBER() OVER (ORDER BY embedding <=> CAST(:embedding AS vector)) AS rank
            FROM document_vectors
            WHERE user_id = :user_id
            LIMIT :candidate_limit
        ),
        keyword_search AS (
            SELECT id,
                   ROW_NUMBER() OVER (ORDER BY content <@> :query) AS rank
            FROM document_vectors
            WHERE user_id = :user_id
            LIMIT :candidate_limit
        )
        SELECT
            d.id,
            d.document_id,
            d.title,
            d.content,
            d.meta_data,
            COALESCE(1.0 / (:rrf_k + v.rank), 0.0)
                + COALESCE(1.0 / (:rrf_k + k.rank), 0.0) AS combined_score
        FROM document_vectors d
        LEFT JOIN vector_search v ON d.id = v.id
        LEFT JOIN keyword_search k ON d.id = k.id
        WHERE (v.id IS NOT NULL OR k.id IS NOT NULL)
          AND d.user_id = :user_id
        ORDER BY combined_score DESC
        LIMIT :fetch_limit OFFSET :offset
        """

        fetch_limit = limit + 1  # one extra to detect has_more

        result = await self.session.execute(
            text(sql),
            {
                "query": query,
                "embedding": embedding_str,
                "user_id": user_id,
                "fetch_limit": fetch_limit,
                "offset": offset,
                "candidate_limit": candidate_limit,
                "rrf_k": rrf_k,
            },
        )
        rows = result.mappings().all()
        has_more = len(rows) > limit
        return rows[:limit], has_more

    async def fuzzy_autocomplete(
        self,
        query: str,
        user_id: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """
        Uses pg_trgm for typo-tolerant prefix/substring matching on titles.
        Requires: CREATE EXTENSION IF NOT EXISTS pg_trgm;
        And a GIN/GiST trgm index on title for performance:
          CREATE INDEX ON document_vectors USING gin(title gin_trgm_ops);
        """
        sql = """
        SELECT title,
               similarity(title, :query) AS score
        FROM document_vectors
        WHERE user_id = :user_id
          AND title % :query
        ORDER BY title <-> :query
        LIMIT :limit
        """
        result = await self.session.execute(
            text(sql),
            {"query": query, "user_id": user_id, "limit": limit},
        )
        return result.mappings().all()

    async def create_document_vector(self, data: dict[str, Any]) -> DocumentVector:
        """Insert a new document vector into the database."""
        doc = DocumentVector(**data)
        self.session.add(doc)
        await self.session.commit()
        await self.session.refresh(doc)
        return doc
