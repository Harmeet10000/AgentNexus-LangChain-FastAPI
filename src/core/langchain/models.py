"""LangChain model initialization and management."""

from functools import lru_cache
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.callbacks import AsyncCallbackHandler

from core.config.settings import settings
from core.config.logging_config import LoggerAdapter

logger = LoggerAdapter(__name__)


class LangChainModels:
    """Container for LangChain models."""
    
    def __init__(self):
        """Initialize LangChain models."""
        self._chat_model: Optional[ChatGoogleGenerativeAI] = None
        self._embedding_model: Optional[GoogleGenerativeAIEmbeddings] = None
        self._vision_model: Optional[ChatGoogleGenerativeAI] = None
    
    @property
    def chat_model(self) -> ChatGoogleGenerativeAI:
        """Get or create chat model."""
        if self._chat_model is None:
            self._chat_model = ChatGoogleGenerativeAI(
                model=settings.gemini_model,
                google_api_key=settings.google_api_key,
                temperature=settings.gemini_temperature,
                max_output_tokens=settings.gemini_max_tokens,
                convert_system_message_to_human=True
            )
            logger.info("Initialized Gemini chat model")
        
        return self._chat_model
    
    @property
    def embedding_model(self) -> GoogleGenerativeAIEmbeddings:
        """Get or create embedding model."""
        if self._embedding_model is None:
            self._embedding_model = GoogleGenerativeAIEmbeddings(
                model=settings.gemini_embedding_model,
                google_api_key=settings.google_api_key
            )
            logger.info("Initialized Gemini embedding model")
        
        return self._embedding_model
    
    @property
    def vision_model(self) -> ChatGoogleGenerativeAI:
        """Get or create vision model."""
        if self._vision_model is None:
            self._vision_model = ChatGoogleGenerativeAI(
                model=settings.gemini_vision_model,
                google_api_key=settings.google_api_key
            )
            logger.info("Initialized Gemini vision model")
        
        return self._vision_model


@lru_cache()
def get_langchain_models() -> LangChainModels:
    """Get cached LangChain models instance."""
    return LangChainModels()


# Convenience functions
def get_chat_model() -> ChatGoogleGenerativeAI:
    """Get chat model."""
    return get_langchain_models().chat_model


def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    """Get embedding model."""
    return get_langchain_models().embedding_model


def get_vision_model() -> ChatGoogleGenerativeAI:
    """Get vision model."""
    return get_langchain_models().vision_model


# Import the existing gemini service for backward compatibility
def get_gemini_service():
    """Get gemini service for backward compatibility."""
    from services.langchain.gemini_service import gemini_service
    return gemini_service