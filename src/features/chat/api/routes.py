"""Chat API routes."""

from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Request

from features.chat.api.schemas import ChatRequest, ChatResponse, MemoryClearResponse
from features.chat.services.chat_service import ChatService, get_chat_service
from core.config.logging_config import LoggerAdapter
from shared.schemas.response import http_success, http_error

logger = LoggerAdapter(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/completions", response_model=ChatResponse)
async def chat_completion(
    request: Request,
    chat_request: ChatRequest,
    service: ChatService = Depends(get_chat_service)
):
    """Generate chat completion."""
    try:
        response = await service.generate_completion(chat_request)
        
        return http_success(
            request,
            message="Chat completion generated successfully",
            data=response
        )
        
    except Exception as e:
        logger.error("Failed to generate chat completion", error=str(e))
        return http_error(request, e, 500)


@router.post("/stream")
async def chat_stream(
    request: Request,
    chat_request: ChatRequest,
    service: ChatService = Depends(get_chat_service)
):
    """Stream chat completion (SSE)."""
    try:
        # Implementation for streaming responses
        return {"message": "Streaming not implemented yet"}
        
    except Exception as e:
        logger.error("Failed to stream chat completion", error=str(e))
        return http_error(request, e, 500)


@router.delete("/memory/{session_id}", response_model=MemoryClearResponse)
async def clear_memory(
    request: Request,
    session_id: str,
    service: ChatService = Depends(get_chat_service)
):
    """Clear conversation memory for a session."""
    try:
        result = await service.clear_memory(session_id)
        
        return http_success(
            request,
            message=f"Memory cleared for session {session_id}",
            data=result
        )
        
    except Exception as e:
        logger.error("Failed to clear memory", error=str(e))
        return http_error(request, e, 500)