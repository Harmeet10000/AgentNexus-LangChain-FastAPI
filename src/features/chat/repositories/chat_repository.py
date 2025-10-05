"""Chat repository implementation."""

from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import Depends

from core.database.mongodb import get_database
from core.config.logging_config import LoggerAdapter

logger = LoggerAdapter(__name__)


class ChatRepository:
    """Repository for chat data operations."""
    
    def __init__(self, db):
        """Initialize chat repository."""
        self.db = db
        self.conversations = db.conversations
        self.sessions = db.chat_sessions
    
    async def save_conversation(self, conversation_data: Dict[str, Any]) -> str:
        """Save conversation to database."""
        try:
            conversation_doc = {
                **conversation_data,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            result = await self.conversations.insert_one(conversation_doc)
            
            logger.info(f"Saved conversation with ID: {result.inserted_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error("Failed to save conversation", error=str(e))
            raise
    
    async def get_conversation_history(
        self, 
        session_id: str, 
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get conversation history for a session."""
        try:
            cursor = self.conversations.find(
                {"session_id": session_id}
            ).sort("created_at", -1).limit(limit)
            
            conversations = await cursor.to_list(length=limit)
            
            logger.info(f"Retrieved {len(conversations)} conversations for session {session_id}")
            return conversations
            
        except Exception as e:
            logger.error("Failed to get conversation history", error=str(e))
            raise
    
    async def clear_session_history(self, session_id: str) -> bool:
        """Clear conversation history for a session."""
        try:
            result = await self.conversations.delete_many({"session_id": session_id})
            
            logger.info(f"Cleared {result.deleted_count} conversations for session {session_id}")
            return result.deleted_count > 0
            
        except Exception as e:
            logger.error("Failed to clear session history", error=str(e))
            raise
    
    async def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a chat session."""
        try:
            pipeline = [
                {"$match": {"session_id": session_id}},
                {
                    "$group": {
                        "_id": "$session_id",
                        "total_messages": {"$sum": 1},
                        "first_message": {"$min": "$created_at"},
                        "last_message": {"$max": "$created_at"}
                    }
                }
            ]
            
            result = await self.conversations.aggregate(pipeline).to_list(length=1)
            
            if result:
                stats = result[0]
                logger.info(f"Retrieved stats for session {session_id}")
                return {
                    "session_id": session_id,
                    "total_messages": stats["total_messages"],
                    "first_message": stats["first_message"],
                    "last_message": stats["last_message"]
                }
            else:
                return {
                    "session_id": session_id,
                    "total_messages": 0,
                    "first_message": None,
                    "last_message": None
                }
                
        except Exception as e:
            logger.error("Failed to get session stats", error=str(e))
            raise


def get_chat_repository(db = Depends(get_database)) -> ChatRepository:
    """Dependency to get chat repository."""
    return ChatRepository(db)