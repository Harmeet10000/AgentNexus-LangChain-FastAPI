
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The raw text query")
    embedding: list[float] | None = Field(
        None, description="Vector embedding of the query"
    )
    limit: int = Field(20, ge=1, le=100)
    offset: int = Field(0, ge=0)


class SearchResultItem(BaseModel):
    id: int
    title: str
    content: str
    combined_score: float | None = None


class SearchResponse(BaseModel):
    items: list[SearchResultItem]
    # Note: Returning exact total_count is an anti-pattern at scale.
    # Use 'has_more' for infinite scroll instead.
    has_more: bool


class AutocompleteResponse(BaseModel):
    suggestions: list[str]
