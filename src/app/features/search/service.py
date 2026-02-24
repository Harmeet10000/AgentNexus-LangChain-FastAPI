from .constants import HYBRID_CANDIDATE_LIMIT, RRF_K
from .dto import (
    AutocompleteResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)
from .repository import SearchRepository


class SearchService:
    def __init__(self, repo: SearchRepository, redis=None):
        self.repo = repo
        self.redis = redis

    async def perform_search(self, request: SearchRequest) -> SearchResponse:
        # Generate embedding if the client didn't provide one
        embedding = await self._generate_embedding(request.query)

        # We dynamically increase the candidate limit based on the offset
        # to ensure RRF has enough data to rank deep pages.
        dynamic_candidate_limit = max(
            HYBRID_CANDIDATE_LIMIT, request.offset + request.limit * 2
        )

        rows = await self.repo.hybrid_search(
            query=request.query,
            embedding=embedding,
            limit=request.limit,
            offset=request.offset,
            candidate_limit=dynamic_candidate_limit,
            rrf_k=RRF_K,
        )

        # Handle "has_more" logic for infinite scrolling
        has_more = len(rows) > request.limit
        if has_more:
            rows = rows[:-1]

        items = [SearchResultItem(**row) for row in rows]
        return SearchResponse(items=items, has_more=has_more)

    async def get_suggestions(self, query: str) -> AutocompleteResponse:
        rows = await self.repo.fuzzy_autocomplete(query, limit=5)
        suggestions = [row["title"] for row in rows]
        return AutocompleteResponse(suggestions=suggestions)

    async def _generate_embedding(self, text: str) -> list[float]:
        # TODO: Call OpenAI/Ollama/Cohere to embed the search query
        return [0.0] * 1536
