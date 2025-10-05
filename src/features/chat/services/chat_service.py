"""Chat service implementation."""

from typing import Dict, Any, Optional
from fastapi import Depends, HTTPException

from features.chat.api.schemas import ChatRequest, ChatResponse, MemoryClearResponse
from features.chat.repositories.chat_repository import ChatRepository, get_chat_repository
from core.config.logging_config import LoggerAdapter
from core.langchain.models import get_gemini_service
from services.langsmith.client import langsmith_service

logger = LoggerAdapter(__name__)


class ChatService:
    """Service for chat operations."""
    
    def __init__(self, chat_repo: ChatRepository):
        """Initialize chat service."""
        self.chat_repo = chat_repo
        self.gemini_service = get_gemini_service()
    
    async def generate_completion(self, request: ChatRequest) -> ChatResponse:
        """Generate chat completion."""
        try:
            # Convert messages to dict format
            messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
            
            # Get LangSmith callbacks
            callbacks = langsmith_service.get_callbacks()
            
            # Generate response
            if request.use_memory and request.session_id:
                # Use conversation memory
                result = await self.gemini_service.generate_with_memory(
                    message=messages[-1]["content"],  # Last message
                    session_id=request.session_id,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    callbacks=callbacks
                )
                
                response_text = result["response"]
                session_id = result["session_id"]
            else:
                # Direct generation
                response_text = await self.gemini_service.generate_response(
                    messages=messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    callbacks=callbacks
                )
                session_id = None
            
            # Save conversation to repository
            conversation_data = {
                "session_id": session_id,
                "messages": messages,
                "response": response_text,
                "model": request.model,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens
            }
            
            await self.chat_repo.save_conversation(conversation_data)
            
            # Log to LangSmith
            await langsmith_service.log_run(
                run_type="chat_completion",
                inputs={"messages": messages, "params": request.dict(exclude={"messages"})},
                outputs={"response": response_text}
            )
            
            return ChatResponse(
                response=response_text,
                model=request.model,
                session_id=session_id
            )
            
        except Exception as e:
            logger.error("Failed to generate chat completion", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    async def clear_memory(self, session_id: str) -> MemoryClearResponse:
        """Clear conversation memory for a session."""
        try:
            success = self.gemini_service.clear_memory(session_id)
            
            if success:
                # Also clear from repository
                await self.chat_repo.clear_session_history(session_id)
                
                return MemoryClearResponse(
                    message=f"Memory cleared for session {session_id}",
                    session_id=session_id
                )
            else:
                raise HTTPException(status_code=404, detail="Session not found")
                
        except Exception as e:
            logger.error("Failed to clear memory", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))


def get_chat_service(
    chat_repo: ChatRepository = Depends(get_chat_repository)
) -> ChatService:
    """Dependency to get chat service."""
    return ChatService(chat_repo)