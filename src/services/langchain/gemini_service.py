"""Gemini LLM service implementation."""

from typing import List, Optional, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.callbacks import AsyncCallbackHandler
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.memory.chat_memory import BaseChatMemory

from src.core.config.settings import settings
from src.core.config.logging_config import LoggerAdapter
from src.core.cache.redis_client import redis_cache

logger = LoggerAdapter(__name__)


class GeminiService:
    """Service for interacting with Google Gemini models."""

    def __init__(self):
        """Initialize Gemini service."""
        self.chat_model = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=settings.gemini_temperature,
            max_output_tokens=settings.gemini_max_tokens,
            convert_system_message_to_human=True,
        )

        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model=settings.gemini_embedding_model,
            google_api_key=settings.google_api_key,
        )

        self.vision_model = ChatGoogleGenerativeAI(
            model=settings.gemini_vision_model, google_api_key=settings.google_api_key
        )

        # Memory stores for conversations
        self._memory_stores: Dict[str, BaseChatMemory] = {}

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        callbacks: Optional[List[AsyncCallbackHandler]] = None,
        use_cache: bool = True,
        cache_key: Optional[str] = None,
    ) -> str:
        """Generate response from Gemini model."""
        try:
            # Check cache if enabled
            if use_cache and cache_key:
                cached_response = await redis_cache.get(
                    f"gemini:response:{cache_key}"
                )
                if cached_response:
                    logger.info("Returning cached response", cache_key=cache_key)
                    return cached_response

            # Convert dict messages to LangChain message objects
            lc_messages = self._convert_to_langchain_messages(messages)

            # Create model with custom parameters if provided
            model = self.chat_model
            if temperature is not None or max_tokens is not None:
                model = ChatGoogleGenerativeAI(
                    model=settings.gemini_model,
                    google_api_key=settings.google_api_key,
                    temperature=temperature or settings.gemini_temperature,
                    max_output_tokens=max_tokens or settings.gemini_max_tokens,
                    convert_system_message_to_human=True,
                )

            # Generate response
            response = await model.ainvoke(lc_messages, callbacks=callbacks)

            response_text = response.content

            # Cache response if enabled
            if use_cache and cache_key:
                await redis_cache.set(
                    f"gemini:response:{cache_key}",
                    response_text,
                    ttl=3600,  # 1 hour cache
                )

            logger.info("Generated response", length=len(response_text))
            return response_text

        except Exception as e:
            logger.error("Failed to generate response", error=str(e))
            raise

    async def generate_embeddings(
        self, texts: List[str], batch_size: int = 100
    ) -> List[List[float]]:
        """Generate embeddings for texts."""
        try:
            all_embeddings = []

            # Process in batches to avoid rate limits
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                embeddings = await self.embedding_model.aembed_documents(batch)
                all_embeddings.extend(embeddings)

                logger.debug(f"Generated embeddings for batch {i//batch_size + 1}")

            logger.info(f"Generated {len(all_embeddings)} embeddings")
            return all_embeddings

        except Exception as e:
            logger.error("Failed to generate embeddings", error=str(e))
            raise

    async def generate_single_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            embedding = await self.embedding_model.aembed_query(text)
            return embedding
        except Exception as e:
            logger.error("Failed to generate single embedding", error=str(e))
            raise

    async def analyze_image(
        self, image_data: bytes, prompt: str, mime_type: str = "image/jpeg"
    ) -> str:
        """Analyze image using Gemini Vision model."""
        try:
            import base64

            # Encode image to base64
            image_base64 = base64.b64encode(image_data).decode("utf-8")

            # Create message with image
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image_base64}"},
                    },
                ]
            )

            # Generate response
            response = await self.vision_model.ainvoke([message])

            logger.info("Analyzed image successfully")
            return response.content

        except Exception as e:
            logger.error("Failed to analyze image", error=str(e))
            raise

    def get_or_create_memory(
        self, session_id: str, memory_type: str = "buffer", max_token_limit: int = 2000
    ) -> BaseChatMemory:
        """Get or create conversation memory for a session."""
        if session_id not in self._memory_stores:
            if memory_type == "summary":
                memory = ConversationSummaryMemory(
                    llm=self.chat_model,
                    max_token_limit=max_token_limit,
                    return_messages=True,
                )
            else:
                memory = ConversationBufferMemory(
                    return_messages=True, memory_key="chat_history"
                )

            self._memory_stores[session_id] = memory
            logger.info(f"Created new {memory_type} memory for session {session_id}")

        return self._memory_stores[session_id]

    async def generate_with_memory(
        self, message: str, session_id: str, memory_type: str = "buffer", **kwargs
    ) -> Dict[str, Any]:
        """Generate response with conversation memory."""
        try:
            # Get or create memory
            memory = self.get_or_create_memory(session_id, memory_type)

            # Get chat history
            chat_history = memory.chat_memory.messages

            # Prepare messages
            messages = []
            for msg in chat_history:
                if isinstance(msg, HumanMessage):
                    messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    messages.append({"role": "assistant", "content": msg.content})
                elif isinstance(msg, SystemMessage):
                    messages.append({"role": "system", "content": msg.content})

            # Add current message
            messages.append({"role": "user", "content": message})

            # Generate response
            response = await self.generate_response(messages, **kwargs)

            # Update memory
            memory.chat_memory.add_user_message(message)
            memory.chat_memory.add_ai_message(response)

            return {
                "response": response,
                "session_id": session_id,
                "message_count": len(memory.chat_memory.messages),
            }

        except Exception as e:
            logger.error("Failed to generate with memory", error=str(e))
            raise

    def clear_memory(self, session_id: str) -> bool:
        """Clear conversation memory for a session."""
        if session_id in self._memory_stores:
            del self._memory_stores[session_id]
            logger.info(f"Cleared memory for session {session_id}")
            return True
        return False

    def _convert_to_langchain_messages(
        self, messages: List[Dict[str, str]]
    ) -> List[BaseMessage]:
        """Convert dictionary messages to LangChain message objects."""
        lc_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))

        return lc_messages


# Singleton instance
gemini_service = GeminiService()
