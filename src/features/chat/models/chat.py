"""Chat database models."""

from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Chat message model."""
    role: str
    content: str
    timestamp: Optional[datetime] = None


class Conversation(BaseModel):
    """Conversation model."""
    id: Optional[str] = Field(default=None, alias="_id")
    session_id: Optional[str] = None
    messages: List[Dict[str, Any]]
    response: str
    model: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        populate_by_name = True


class ChatSession(BaseModel):
    """Chat session model."""
    id: Optional[str] = Field(default=None, alias="_id")
    session_id: str
    user_id: Optional[str] = None
    title: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    is_active: bool = True
    
    class Config:
        populate_by_name = True