"""Base schemas for common patterns."""

from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class BaseEntity(BaseModel):
    """Base entity with common fields."""
    id: Optional[str] = Field(default=None, alias="_id")
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        populate_by_name = True


class PaginationParams(BaseModel):
    """Pagination parameters."""
    page: int = Field(default=1, ge=1)
    size: int = Field(default=20, ge=1, le=100)
    
    @property
    def skip(self) -> int:
        """Calculate skip value for database queries."""
        return (self.page - 1) * self.size


class PaginatedResponse(BaseModel):
    """Paginated response wrapper."""
    items: list
    total: int
    page: int
    size: int
    pages: int
    
    @classmethod
    def create(
        cls,
        items: list,
        total: int,
        pagination: PaginationParams
    ) -> "PaginatedResponse":
        """Create paginated response."""
        pages = (total + pagination.size - 1) // pagination.size
        
        return cls(
            items=items,
            total=total,
            page=pagination.page,
            size=pagination.size,
            pages=pages
        )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    version: str
    environment: str
    services: Dict[str, str] = Field(default_factory=dict)


class ErrorDetail(BaseModel):
    """Error detail model."""
    field: str
    message: str
    code: Optional[str] = None