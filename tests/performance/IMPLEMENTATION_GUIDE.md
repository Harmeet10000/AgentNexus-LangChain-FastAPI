# Implementation Guide

This guide provides detailed implementation examples for all major components of the LangChain FastAPI Production template.

## Table of Contents

1. [LangChain with Gemini Integration](#langchain-with-gemini-integration)
2. [Pinecone Vector Store](#pinecone-vector-store)
3. [LangGraph Workflows](#langgraph-workflows)
4. [LangSmith Monitoring](#langsmith-monitoring)
5. [Document Processing with Docling](#document-processing-with-docling)
6. [Web Crawling with Crawl4AI](#web-crawling-with-crawl4ai)
7. [Complete RAG Pipeline](#complete-rag-pipeline)
8. [API Endpoints](#api-endpoints)

## LangChain with Gemini Integration

### Service Implementation (`src/services/langchain/gemini_service.py`)

```python
"""Gemini LLM service implementation."""

from typing import List, Optional, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.callbacks import AsyncCallbackHandler
from src.core.config.settings import settings
from src.core.config.logging_config import LoggerAdapter

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
            convert_system_message_to_human=True
        )
        
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model=settings.gemini_embedding_model,
            google_api_key=settings.google_api_key
        )
        
        self.vision_model = ChatGoogleGenerativeAI(
            model=settings.gemini_vision_model,
            google_api_key=settings.google_api_key
        )
    
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        callbacks: Optional[List[AsyncCallbackHandler]] = None
    ) -> str:
        """Generate response from Gemini model."""
        try:
            # Convert dict messages to LangChain message objects
            lc_messages = self._convert_to_langchain_messages(messages)
            
            # Override temperature if provided
            if temperature is not None:
                self.chat_model.temperature = temperature
            
            # Generate response
            response = await self.chat_model.ainvoke(
                lc_messages,
                callbacks=callbacks
            )
            
            return response.content
            
        except Exception as e:
            logger.error("Failed to generate response", error=str(e))
            raise
    
    async def generate_embeddings(
        self,
        texts: List[str]
    ) -> List[List[float]]:
        """Generate embeddings for texts."""
        try:
            embeddings = await self.embedding_model.aembed_documents(texts)
            return embeddings
        except Exception as e:
            logger.error("Failed to generate embeddings", error=str(e))
            raise
    
    async def analyze_image(
        self,
        image_data: bytes,
        prompt: str
    ) -> str:
        """Analyze image using Gemini Vision model."""
        try:
            # Implementation for image analysis
            # This would require proper image encoding
            pass
        except Exception as e:
            logger.error("Failed to analyze image", error=str(e))
            raise
    
    def _convert_to_langchain_messages(
        self,
        messages: List[Dict[str, str]]
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
```

## Pinecone Vector Store

### Service Implementation (`src/services/pinecone/client.py`)

```python
"""Pinecone vector store service."""

import os
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from langchain.vectorstores import Pinecone as LangchainPinecone
from src.core.config.settings import settings
from src.core.config.logging_config import LoggerAdapter

logger = LoggerAdapter(__name__)

pinecone_client: Optional[Pinecone] = None
pinecone_index = None


def initialize_pinecone():
    """Initialize Pinecone client and index."""
    global pinecone_client, pinecone_index
    
    try:
        logger.info("Initializing Pinecone")
        
        pinecone_client = Pinecone(
            api_key=settings.pinecone_api_key,
            environment=settings.pinecone_environment
        )
        
        # Create index if it doesn't exist
        if settings.pinecone_index_name not in pinecone_client.list_indexes().names():
            pinecone_client.create_index(
                name=settings.pinecone_index_name,
                dimension=settings.pinecone_dimension,
                metric=settings.pinecone_metric,
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-west-2'
                )
            )
        
        pinecone_index = pinecone_client.Index(settings.pinecone_index_name)
        logger.info("Pinecone initialized successfully")
        
    except Exception as e:
        logger.error("Failed to initialize Pinecone", error=str(e))
        raise


class VectorStoreService:
    """Service for vector store operations."""
    
    def __init__(self):
        """Initialize vector store service."""
        from src.services.langchain.gemini_service import gemini_service
        self.embedding_model = gemini_service.embedding_model
        self.vectorstore = None
        
        if pinecone_index:
            self.vectorstore = LangchainPinecone(
                pinecone_index,
                self.embedding_model.embed_query,
                "text"
            )
    
    async def add_documents(
        self,
        documents: List[Dict[str, Any]],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """Add documents to vector store."""
        try:
            if not self.vectorstore:
                raise RuntimeError("Vector store not initialized")
            
            texts = [doc.get("content", "") for doc in documents]
            metadatas = metadata or [doc.get("metadata", {}) for doc in documents]
            
            ids = await self.vectorstore.aadd_texts(
                texts=texts,
                metadatas=metadatas
            )
            
            logger.info(f"Added {len(ids)} documents to vector store")
            return ids
            
        except Exception as e:
            logger.error("Failed to add documents", error=str(e))
            raise
    
    async def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform similarity search."""
        try:
            if not self.vectorstore:
                raise RuntimeError("Vector store not initialized")
            
            results = await self.vectorstore.asimilarity_search_with_score(
                query=query,
                k=k,
                filter=filter
            )
            
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score
                }
                for doc, score in results
            ]
            
        except Exception as e:
            logger.error("Failed to perform similarity search", error=str(e))
            raise
    
    async def delete_by_ids(self, ids: List[str]) -> bool:
        """Delete vectors by IDs."""
        try:
            if not pinecone_index:
                raise RuntimeError("Pinecone index not initialized")
            
            pinecone_index.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} vectors")
            return True
            
        except Exception as e:
            logger.error("Failed to delete vectors", error=str(e))
            return False
```

## LangGraph Workflows

### Workflow Implementation (`src/graphs/workflows/rag_workflow.py`)

```python
"""RAG workflow using LangGraph."""

from typing import TypedDict, List, Annotated, Sequence
from langgraph.graph import Graph, StateGraph
from langgraph.prebuilt import ToolExecutor
from langchain.schema import BaseMessage
import operator

class RAGState(TypedDict):
    """State for RAG workflow."""
    query: str
    context: List[str]
    messages: Annotated[Sequence[BaseMessage], operator.add]
    answer: str
    sources: List[Dict]


def create_rag_workflow():
    """Create RAG workflow graph."""
    
    workflow = StateGraph(RAGState)
    
    async def retrieve_context(state: RAGState) -> RAGState:
        """Retrieve relevant context from vector store."""
        from src.services.pinecone.client import VectorStoreService
        
        vector_service = VectorStoreService()
        results = await vector_service.similarity_search(
            query=state["query"],
            k=5
        )
        
        state["context"] = [r["content"] for r in results]
        state["sources"] = results
        return state
    
    async def generate_answer(state: RAGState) -> RAGState:
        """Generate answer using context."""
        from src.services.langchain.gemini_service import gemini_service
        
        # Create prompt with context
        context_str = "\n\n".join(state["context"])
        prompt = f"""
        Based on the following context, answer the question.
        
        Context:
        {context_str}
        
        Question: {state["query"]}
        
        Answer:
        """
        
        answer = await gemini_service.generate_response([
            {"role": "user", "content": prompt}
        ])
        
        state["answer"] = answer
        return state
    
    async def validate_answer(state: RAGState) -> RAGState:
        """Validate and refine answer."""
        # Add validation logic here
        return state
    
    # Add nodes
    workflow.add_node("retrieve", retrieve_context)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("validate", validate_answer)
    
    # Add edges
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "validate")
    workflow.set_finish_point("validate")
    
    return workflow.compile()


# Create workflow instance
rag_workflow = create_rag_workflow()
```

## Document Processing with Docling

### Service Implementation (`src/services/docling/processor.py`)

```python
"""Document processing service using Docling."""

from typing import List, Dict, Any, Optional
from pathlib import Path
import tempfile
from docling import Document
from docling.datamodel import InputFormat
from src.core.config.logging_config import LoggerAdapter

logger = LoggerAdapter(__name__)


class DocumentProcessor:
    """Service for processing documents."""
    
    async def process_document(
        self,
        file_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[Dict[str, Any]]:
        """Process document and extract content."""
        try:
            # Load document
            doc = Document.from_file(file_path)
            
            # Extract text
            full_text = doc.export_to_text()
            
            # Extract metadata
            metadata = {
                "filename": Path(file_path).name,
                "page_count": len(doc.pages) if hasattr(doc, 'pages') else 1,
                "format": Path(file_path).suffix
            }
            
            # Chunk the document
            chunks = self._chunk_text(
                full_text,
                chunk_size,
                chunk_overlap
            )
            
            # Create chunk documents
            documents = []
            for i, chunk in enumerate(chunks):
                documents.append({
                    "content": chunk,
                    "metadata": {
                        **metadata,
                        "chunk_index": i,
                        "chunk_total": len(chunks)
                    }
                })
            
            logger.info(f"Processed document: {metadata['filename']}, {len(chunks)} chunks")
            return documents
            
        except Exception as e:
            logger.error("Failed to process document", error=str(e))
            raise
    
    def _chunk_text(
        self,
        text: str,
        chunk_size: int,
        overlap: int
    ) -> List[str]:
        """Chunk text with overlap."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
        
        return chunks
    
    async def extract_tables(self, file_path: str) -> List[Dict]:
        """Extract tables from document."""
        try:
            doc = Document.from_file(file_path)
            tables = doc.export_to_tables()
            return tables
        except Exception as e:
            logger.error("Failed to extract tables", error=str(e))
            return []
    
    async def extract_images(self, file_path: str) -> List[bytes]:
        """Extract images from document."""
        try:
            doc = Document.from_file(file_path)
            images = doc.export_to_images()
            return images
        except Exception as e:
            logger.error("Failed to extract images", error=str(e))
            return []
```

## Complete RAG Pipeline

### API Endpoint (`src/api/endpoints/rag.py`)

```python
"""RAG API endpoints."""

from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
from src.services.docling.processor import DocumentProcessor
from src.services.pinecone.client import VectorStoreService
from src.graphs.workflows.rag_workflow import rag_workflow
from src.core.config.logging_config import LoggerAdapter

logger = LoggerAdapter(__name__)
router = APIRouter(prefix="/rag", tags=["RAG"])


class RAGQuery(BaseModel):
    """RAG query request model."""
    query: str
    document_ids: Optional[List[str]] = None
    top_k: int = 5


class RAGResponse(BaseModel):
    """RAG response model."""
    answer: str
    sources: List[Dict]
    confidence: float


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: int = 1000,
    chunk_overlap: int = 200
):
    """Upload and process document for RAG."""
    try:
        # Save uploaded file
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Process document
        processor = DocumentProcessor()
        chunks = await processor.process_document(
            tmp_path,
            chunk_size,
            chunk_overlap
        )
        
        # Add to vector store
        vector_service = VectorStoreService()
        ids = await vector_service.add_documents(chunks)
        
        # Clean up
        import os
        os.unlink(tmp_path)
        
        return {
            "message": "Document processed successfully",
            "document_id": file.filename,
            "chunks": len(chunks),
            "vector_ids": ids
        }
        
    except Exception as e:
        logger.error("Failed to upload document", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=RAGResponse)
async def query_rag(request: RAGQuery):
    """Query RAG system."""
    try:
        # Run RAG workflow
        result = await rag_workflow.ainvoke({
            "query": request.query,
            "context": [],
            "messages": [],
            "answer": "",
            "sources": []
        })
        
        return RAGResponse(
            answer=result["answer"],
            sources=result["sources"],
            confidence=0.95  # Calculate actual confidence
        )
        
    except Exception as e:
        logger.error("Failed to query RAG", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
```

## Web Crawling with Crawl4AI

### Service Implementation (`src/services/crawl4ai/crawler.py`)

```python
"""Web crawling service using Crawl4AI."""

from typing import List, Dict, Any, Optional
from crawl4ai import AsyncWebCrawler
from crawl4ai.extraction_strategy import JsonExtractionStrategy, LLMExtractionStrategy
from src.core.config.settings import settings
from src.core.config.logging_config import LoggerAdapter

logger = LoggerAdapter(__name__)


class WebCrawler:
    """Service for web crawling and content extraction."""
    
    def __init__(self):
        """Initialize web crawler."""
        self.crawler = AsyncWebCrawler(
            headless=settings.crawl4ai_headless,
            timeout=settings.crawl4ai_timeout,
            user_agent=settings.crawl4ai_user_agent
        )
    
    async def crawl_url(
        self,
        url: str,
        max_depth: int = 1,
        extract_links: bool = True,
        extract_images: bool = False
    ) -> Dict[str, Any]:
        """Crawl a URL and extract content."""
        try:
            async with self.crawler as crawler:
                result = await crawler.arun(
                    url=url,
                    word_count_threshold=10,
                    extraction_strategy=JsonExtractionStrategy(
                        schema={
                            "title": "string",
                            "content": "string",
                            "links": ["string"],
                            "images": ["string"]
                        }
                    ) if extract_links or extract_images else None
                )
                
                return {
                    "url": url,
                    "title": result.metadata.get("title", ""),
                    "content": result.markdown,
                    "links": result.links if extract_links else [],
                    "images": result.images if extract_images else [],
                    "metadata": result.metadata
                }
                
        except Exception as e:
            logger.error("Failed to crawl URL", url=url, error=str(e))
            raise
    
    async def crawl_sitemap(
        self,
        sitemap_url: str,
        max_pages: int = 10
    ) -> List[Dict[str, Any]]:
        """Crawl pages from sitemap."""
        try:
            # Implementation for sitemap crawling
            pages = []
            # Parse sitemap and crawl pages
            return pages
        except Exception as e:
            logger.error("Failed to crawl sitemap", error=str(e))
            raise
```

## API Router Configuration

### Main Router (`src/api/router.py`)

```python
"""Main API router."""

from fastapi import APIRouter
from src.api.endpoints import chat, rag, documents, crawl

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(chat.router)
api_router.include_router(rag.router)
api_router.include_router(documents.router)
api_router.include_router(crawl.router)
```

## Next Steps

1. **Add Authentication**: Implement JWT-based authentication in `src/core/security/`
2. **Add More Chains**: Create conversation chains, structured output chains
3. **Implement Caching**: Add Redis caching for responses
4. **Add Tests**: Create comprehensive test suite
5. **Add Monitoring**: Implement Prometheus metrics and health checks

This implementation guide provides the foundation for a production-ready LangChain FastAPI application. Each component can be extended and customized based on specific requirements.