from pydantic import BaseModel, ConfigDict, Field

# Global config for response models to keep them lean
fast_model_config = ConfigDict(from_attributes=True)


class SearchRequest(BaseModel):
    # Use strict=True to skip coercion logic
    query: str = Field(strict=True, min_length=1)
    # Large lists are slow; consider using np.ndarray if performance stalls
    embedding: list[float] | None = None
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class SearchResultItem(BaseModel):
    model_config = fast_model_config
    id: int
    title: str
    content: str
    combined_score: float | None = None


class SearchResponse(BaseModel):
    model_config = fast_model_config
    items: list[SearchResultItem]
    has_more: bool


class DocumentVectorCreate(BaseModel):
    # Fixed length strings are slightly faster to validate
    user_id: str = Field(min_length=1, max_length=100)
    document_id: str = Field(min_length=1, max_length=100)
    title: str = Field(min_length=1, max_length=500)
    content: str
    embedding: list[float] | None = None
    vector_id: str | None = Field(default=None, max_length=100)
    metadata: dict | None = None


class DocumentVectorResponse(BaseModel):
    model_config = fast_model_config
    id: int
    user_id: str
    document_id: str
    title: str
    content: str
    vector_id: str | None
    metadata: dict | None = Field(alias="meta_data", default=None)
