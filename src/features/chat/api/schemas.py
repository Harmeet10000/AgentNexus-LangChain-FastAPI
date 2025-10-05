"""Chat API schemas."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Chat message model."""
    role: str = Field(..., description="Message role: user, assistant, or system")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Chat completion request model."""
    messages: List[ChatMessage]
    model: Optional[str] = Field(default="gemini-pro", description="Model to use")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=2048, ge=1, le=8192)
    session_id: Optional[str] = Field(default=None, description="Session ID for conversation memory")
    use_memory: bool = Field(default=False, description="Use conversation memory")


class ChatResponse(BaseModel):
    """Chat completion response model."""
    response: str
    model: str
    usage: Optional[Dict[str, int]] = None
    session_id: Optional[str] = None


class MemoryClearResponse(BaseModel):
    """Memory clear response model."""
    message: str
    session_id: str