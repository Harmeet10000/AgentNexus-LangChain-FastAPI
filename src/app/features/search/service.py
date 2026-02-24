from datetime import UTC, datetime

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.features.search.dto import (
    AutocompleteResponse,
    DocumentVectorCreate,
    DocumentVectorResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)
from app.features.search.repository import SearchRepository

from .constants import HYBRID_CANDIDATE_LIMIT, RRF_K


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

        rows, has_more = await self.repo.hybrid_search(
            query=request.query,
            embedding=embedding,
            user_id="user_001",  # TODO: Support multi-user by passing user_id
            limit=request.limit,
            offset=request.offset,
            candidate_limit=dynamic_candidate_limit,
            rrf_k=RRF_K,
        )

        # Convert Row objects to dicts for Pydantic validation
        items = [SearchResultItem(**dict(row)) for row in rows]
        return SearchResponse(items=items, has_more=has_more)

    async def get_suggestions(self, query: str) -> AutocompleteResponse:
        rows = await self.repo.fuzzy_autocomplete(query, user_id="user_001", limit=5)
        suggestions = [row["title"] for row in rows]
        return AutocompleteResponse(suggestions=suggestions)

    async def _generate_embedding(self, text: str) -> list[float]:
        """Generate embeddings using Google Generative AI."""

        embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            task_type="retrieval_document",
            output_dimensionality=1536,
        )
        return await embeddings.aembed_query(text)

    async def create_document(
        self, data: DocumentVectorCreate
    ) -> DocumentVectorResponse:
        """Create a new document vector entry."""
        doc_data = data.model_dump()
        doc_data["created_at"] = datetime.now(tz=UTC)
        doc_data["updated_at"] = datetime.now(tz=UTC)
        doc_data["meta_data"] = doc_data.pop("metadata", None)

        # Generate embedding if not provided
        if not doc_data.get("embedding"):
            doc_data["embedding"] = await self._generate_embedding(
                f"{data.title} {data.content}"
            )
        # logger.info(
        #     f"Creating document vector for document_id={data.document_id} with embedding size={len(doc_data['embedding'])}"
        # )

        doc = await self.repo.create_document_vector(doc_data)
        return DocumentVectorResponse.model_validate(doc)
