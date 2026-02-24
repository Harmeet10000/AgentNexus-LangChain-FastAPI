from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


class SearchRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def hybrid_search(
        self,
        query: str,
        embedding: list[float],
        limit: int,
        offset: int,
        candidate_limit: int,
        rrf_k: int,
    ) -> list[dict[str, Any]]:
        """
        Executes a Reciprocal Rank Fusion (RRF) query.
        - pg_textsearch uses <@> (returns negative BM25 score, so ASC order is best match)
        - pgvectorscale uses <=> (returns cosine distance, so ASC order is best match)
        """
        sql = """
        WITH vector_search AS (
            SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> :embedding::vector) as rank
            FROM documents
            LIMIT :candidate_limit
        ),
        keyword_search AS (
            SELECT id, ROW_NUMBER() OVER (ORDER BY content <@> :query) as rank
            FROM documents
            LIMIT :candidate_limit
        )
        SELECT
            d.id, d.title, d.content,
            COALESCE(1.0 / (:rrf_k + v.rank), 0.0) + COALESCE(1.0 / (:rrf_k + k.rank), 0.0) as combined_score
        FROM documents d
        LEFT JOIN vector_search v ON d.id = v.id
        LEFT JOIN keyword_search k ON d.id = k.id
        WHERE v.id IS NOT NULL OR k.id IS NOT NULL
        ORDER BY combined_score DESC
        LIMIT :limit OFFSET :offset;
        """

        result = await self.session.execute(
            text(sql),
            {
                "query": query,
                "embedding": str(
                    embedding
                ),  # pgvector expects a string representation of the array
                "limit": limit + 1,  # Fetch 1 extra to determine 'has_more'
                "offset": offset,
                "candidate_limit": candidate_limit,
                "rrf_k": rrf_k,
            },
        )
        return result.mappings().all()

    async def fuzzy_autocomplete(self, query: str, limit: int) -> list[dict[str, Any]]:
        """
        Uses pg_trgm for ultra-fast, typo-tolerant substring matching.
        """
        sql = """
        SELECT title, similarity(title, :query) as score
        FROM documents
        WHERE title % :query
        ORDER BY title <-> :query
        LIMIT :limit;
        """
        result = await self.session.execute(text(sql), {"query": query, "limit": limit})
        return result.mappings().all()
