# LangChain FastAPI Production - AI Development Guidelines

<Info>
    This document serves as the authoritative guide for AI agents and developers
    working on this project. It covers coding standards, architectural patterns,
    and best practices specific to this codebase.
</Info>

## Project Overview

**Project**: langchain-fastapi-production
**Language**: Python 3.12
**Framework**: FastAPI + LangChain
**Architecture Style**: Modular, feature-driven with emphasis on clean code principles

---

## Core Principles

### The Three Pillars

1. **Functional Programming**: Favor pure functions, immutability, and composition
2. **DRY (Don't Repeat Yourself)**: Eliminate code duplication through abstraction
3. **KISS (Keep It Simple, Stupid)**: Prefer simple, straightforward solutions

### Key Tenets

-   **Immutability First**: Prefer immutable data structures and pure functions
-   **Type Safety**: Use type hints everywhere; strict mypy compatibility required
-   **No Module-Level Mutable State**: Avoid global variables; use dependency injection instead
-   **Async-First**: All I/O should be asynchronous
-   **Explicit Over Implicit**: Make dependencies and behavior clear

---

## FastAPI Best Practices

### Router Organization

```python
# ✅ CORRECT: APIRouter per feature with clear prefix
from fastapi import APIRouter, Depends

router = APIRouter(prefix="/documents", tags=["documents"])

@router.get("/", response_model=List[DocumentOut])
async def get_documents(
    user_id: str = Depends(get_user_id),
    service: DocumentService = Depends(get_document_service)
) -> List[DocumentOut]:
    """Thin endpoint - delegates to service."""
    return await service.list_documents_for_user(user_id)
```

**Key Rules**:

-   One APIRouter per feature/domain
-   Mount all routers in a single location (typically `src/api/routes.py`)
-   Keep endpoints thin (5-15 lines max)
-   Delegate business logic to service layer

### Dependency Injection Pattern

```python
# ✅ CORRECT: Factory function + Depends for DI
from typing import Callable
from fastapi import Depends

def get_document_service() -> DocumentService:
    """Factory function - called per request, not at import time."""
    return DocumentService()

@router.get("/documents")
async def list_documents(
    service: DocumentService = Depends(get_document_service)
) -> List[DocumentOut]:
    return await service.list_documents()

# ❌ WRONG: Global singleton at import time
service = DocumentService()  # BAD: Blocks startup, untestable

@router.get("/documents")
async def list_documents() -> List[DocumentOut]:
    return await service.list_documents()
```

### Pydantic Models (v2)

```python
# ✅ CORRECT: Request/Response models with validation
from pydantic import BaseModel, Field, field_validator

class DocumentCreate(BaseModel):
    """Request model - Pydantic v2 validation."""
    title: str = Field(..., min_length=1, max_length=200)
    content: str
    tags: list[str] | None = None

    @field_validator('title')
    @classmethod
    def title_not_empty(cls, v: str) -> str:
        """Pure validator function."""
        if not v.strip():
            raise ValueError('Title cannot be empty')
        return v.strip()

class DocumentOut(BaseModel):
    """Response model."""
    id: str
    title: str
    content: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

# ✅ Use .model_dump() in Pydantic v2
document_dict = document.model_dump(exclude={'id'})
```

### Error Handling

```python
# ✅ CORRECT: Custom exceptions with middleware handling
from fastapi import HTTPException, status

class NotFoundException(HTTPException):
    """Domain-specific exception."""
    def __init__(self, resource: str, identifier: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{resource} with ID {identifier} not found"
        )

class ValidationException(HTTPException):
    """Validation error."""
    def __init__(self, message: str):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=message
        )

# In endpoint:
@router.get("/documents/{doc_id}")
async def get_document(doc_id: str, service: DocumentService = Depends()) -> DocumentOut:
    document = await service.get_by_id(doc_id)
    if document is None:
        raise NotFoundException("Document", doc_id)
    return document
```

### Async I/O

```python
# ✅ CORRECT: Async handlers and async I/O
@router.get("/search")
async def search(
    query: str,
    db: AsyncSession = Depends(get_db),
    search_service: SearchService = Depends()
) -> list[SearchResult]:
    """All operations are async."""
    results = await search_service.search(query)
    return results

# ✅ CORRECT: Running blocking code in thread pool
import asyncio

@router.post("/process")
async def process_file(file: UploadFile = File(...)):
    """Blocking operation runs in thread pool."""
    def _process_blocking(data: bytes) -> dict:
        # CPU-intensive or blocking operation
        return expensive_computation(data)

    content = await file.read()
    result = await asyncio.to_thread(_process_blocking, content)
    return {"status": "processed", "result": result}

# ❌ WRONG: Blocking in async function
@router.post("/process")
async def process_file(file: UploadFile = File(...)):
    content = file.read()  # BLOCKS! Use await asyncio.to_thread instead
    return process_data(content)
```

### Package Imports (**init**.py Standardization)

Every package directory must have a properly structured `__init__.py` file that exports public APIs. This enables clean, predictable imports across the codebase.

#### Pattern

```python
# src/app/utils/__init__.py
"""Utility modules for the application."""

from .apiFeatures import APIFeatures
from .exceptions import APIException, NotFound, ValidationException
from .httpResponse import APIResponse, APIErrorResponse
from .logger import logger
from .quicker import get_first_available

__all__ = [
    'APIFeatures',
    'APIException',
    'NotFound',
    'ValidationException',
    'APIResponse',
    'APIErrorResponse',
    'logger',
    'get_first_available',
]
```

#### Importing from **init**.py

Instead of importing directly from submodules, always import from the package:

```python
# ✅ CORRECT: Import from package __init__.py
from src.app.utils import logger, APIException, APIResponse
from src.app.connections import get_db, CacheManager
from src.app.core import Settings, lifespan

# ❌ WRONG: Direct imports from submodules
from src.app.utils.logger import logger  # Use package instead
from src.app.utils.exceptions import APIException  # Use package instead
from src.app.connections.postgres import get_db  # Use package instead
```

#### Rules for **init**.py Files

1. **Always include docstrings**: Describe the package purpose
2. **Import all public exports**: Any class, function, or constant that external code should access
3. **Maintain **all****: Explicitly list all public exports in `__all__`
4. **Handle empty modules**: For modules under development, use placeholder comments:

    ```python
    # src/app/shared/rag/__init__.py
    """RAG (Retrieval-Augmented Generation) utilities."""

    # Import from submodules when they have content
    # from .retriever import ...
    # from .vectorstore import ...

    __all__ = []
    ```

5. **Organize by layer**: Group imports logically (connections, core, utils, etc.)

#### Directory Structure for Imports

```
src/app/
├── __init__.py                 # Exports: app
├── connections/
│   ├── __init__.py            # Exports: get_db, CacheManager, VectorStoreService, etc.
│   ├── postgres.py
│   ├── redis.py
│   ├── mongodb.py
│   └── pinecone.py
├── core/
│   ├── __init__.py            # Exports: Settings, lifespan, setup_signal_handlers
│   ├── settings.py
│   ├── lifespan.py
│   └── signals.py
├── utils/
│   ├── __init__.py            # Exports: logger, APIException, APIResponse, etc.
│   ├── logger.py
│   ├── exceptions.py
│   ├── httpResponse.py
│   └── quicker.py
├── middleware/
│   ├── __init__.py            # Exports: middleware functions and handlers
│   ├── global_exception_handler.py
│   └── server_middleware.py
├── features/
│   ├── __init__.py            # Lists all features
│   ├── documents/
│   │   ├── __init__.py        # Exports: router, schema, service
│   │   ├── router.py
│   │   ├── schema.py
│   │   └── service.py
│   └── agents/
│       ├── __init__.py        # Exports: router, schema, service
│       ├── router.py
│       ├── schema.py
│       └── service.py
└── shared/
    ├── __init__.py            # Lists shared modules
    ├── agents/
    │   ├── __init__.py        # Exports: agent classes, factory, registry
    │   ├── base_agent.py
    │   ├── agent_factory.py
    │   └── agent_registry.py
    ├── langchain/
    │   ├── __init__.py        # Exports: LangChain utilities
    │   └── agents.py
    └── rag/
        ├── __init__.py        # Exports: RAG utilities
        └── ...
```

#### Example: Using Properly Exported Packages

```python
# src/app/main.py
from src.app.core import Settings, lifespan
from src.app.connections import get_db, connect_to_mongodb, connect_to_redis
from src.app.middleware import (
    global_exception_handler,
    correlation_middleware,
)
from src.app.utils import logger, APIException

# Feature routes
from src.app.features.documents import router as documents_router
from src.app.features.agents import router as agents_router

app = FastAPI(...)

# Clean, predictable imports with full IDE autocomplete support
```

---

## Coding Guidelines

### Function Design

```python
# ✅ CORRECT: Small, pure functions with clear contracts
from typing import Iterator

def filter_active_users(users: list[User]) -> list[User]:
    """Pure function - no side effects."""
    return [u for u in users if u.is_active]

def calculate_total_cost(items: list[CartItem], tax_rate: float) -> float:
    """Pure function with deterministic output."""
    subtotal = sum(item.price * item.quantity for item in items)
    return subtotal * (1 + tax_rate)

# ✅ CORRECT: Composition of smaller functions
async def process_order(order_id: str, service: OrderService = Depends()) -> OrderResponse:
    """Composed from smaller, testable functions."""
    order = await service.get_order(order_id)
    validate_order_status(order)
    enriched_order = await enrich_with_totals(order, service)
    result = await service.save_order(enriched_order)
    return result

# ❌ WRONG: Long function with multiple responsibilities
async def process_order(order_id: str):
    # Get order
    # Validate
    # Calculate totals
    # Update database
    # Send email
    # Log metrics
    # ... too many responsibilities!
    pass
```

### Type Hints

```python
# ✅ CORRECT: Comprehensive type hints
from typing import Optional, Callable, TypeVar
from datetime import datetime

T = TypeVar('T')

async def fetch_resource(
    resource_id: str,
    retry_count: int = 3
) -> Optional[Resource]:
    """Explicit return type."""
    pass

def create_handler(
    service: ServiceType,
    logger: Logger
) -> Callable[[Request], Awaitable[Response]]:
    """Higher-order function with complete types."""
    async def handler(request: Request) -> Response:
        pass
    return handler

# ❌ WRONG: Missing type hints
async def fetch_resource(resource_id, retry_count=3):
    """No type information for AI or type checker."""
    pass
```

### Comparison and None Checks

```python
# ✅ CORRECT: Use 'is' and 'is not' for None/boolean
value: Optional[str] = get_value()
if value is None:
    handle_missing()

is_active: bool = check_status()
if is_active is True:
    proceed()

if is_active is False:
    skip()

# ❌ WRONG: Using == for None/boolean
if value == None:  # Use 'is None'
    pass

if is_active == True:  # Use 'is True'
    pass
```

### Comprehensions

```python
# ✅ CORRECT: Use comprehensions instead of loops
# List comprehension
active_users = [u for u in users if u.is_active]

# Dict comprehension
user_map = {u.id: u.name for u in users}

# Set comprehension
unique_ids = {u.id for u in users}

# Generator expression (for memory efficiency)
large_result = (process(item) for item in huge_list)

# ❌ WRONG: Manual loops
active_users = []
for u in users:
    if u.is_active:
        active_users.append(u)
```

### Context Managers

```python
# ✅ CORRECT: Use 'with' for resource management
from contextlib import asynccontextmanager

async def get_db_connection() -> AsyncGenerator:
    """Context manager for database connection."""
    connection = await create_connection()
    try:
        yield connection
    finally:
        await connection.close()

# Usage with resource cleanup
async with get_db_connection() as db:
    results = await db.query("SELECT * FROM users")

# ✅ CORRECT: Multiple resources
with open('input.txt') as input_file, open('output.txt', 'w') as output_file:
    for line in input_file:
        output_file.write(transform(line))

# ❌ WRONG: Manual resource management
connection = await create_connection()
try:
    results = await connection.query()
finally:
    await connection.close()
```

### Exception Handling

```python
# ✅ CORRECT: Try-except for expected exceptions
from src.core.exceptions import NotFound

async def get_user(user_id: str) -> User:
    """Handle expected exceptions."""
    try:
        user = await db.get_user(user_id)
        return user
    except DatabaseError as e:
        logger.error(
        f"AUTH_FAILED: {e}",
        user_id="user_id",
    )
        raise NotFound(f"User {user_id} not found")

# ✅ CORRECT: Context manager for exception handling
from contextlib import suppress

def safe_close(resource):
    """Silently ignore close errors."""
    with suppress(Exception):
        resource.close()

# ❌ WRONG: Using if-else for exception cases
def get_user(user_id: str) -> User:
    if not user_exists(user_id):
        raise NotFound()
    # Better to try the operation and handle exception
```

---

## Design Patterns

### Factory Pattern

```python
# ✅ CORRECT: Factory for creating configured clients
from typing import Callable
from src.core.config import get_settings

def make_redis_client(url: str) -> Callable[[], RedisClient]:
    """
    Factory function returns a callable that creates clients.
    Initialization (connection) happens separately at startup.
    """
    def factory() -> RedisClient:
        # Create and configure, but don't connect yet
        return RedisClient.from_url(url)
    return factory

# In app startup:
@app.on_event("startup")
async def startup():
    settings = get_settings()
    redis_factory = make_redis_client(settings.REDIS_URL)
    redis_client = redis_factory()
    await redis_client.connect()  # Connect at startup, not import time

# Dependency:
async def get_redis(client: RedisClient = Depends(lambda: redis_client)) -> RedisClient:
    return client
```

### Adapter Pattern

```python
# ✅ CORRECT: Adapter wrapping third-party client
from typing import Any
import asyncio

class VectorDBAdapter:
    """Adapter for vector database - exposes only needed methods."""

    def __init__(self, client: Any):
        self._client = client

    async def upsert(self, items: list[dict[str, Any]]) -> bool:
        """Adapt sync API to async."""
        return await asyncio.to_thread(
            self._client.upsert,
            items
        )

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 10
    ) -> list[dict[str, Any]]:
        """Search with type safety."""
        results = await asyncio.to_thread(
            self._client.search,
            query_vector,
            top_k
        )
        return results

# Usage:
@router.post("/vectors/search")
async def search_vectors(
    query: SearchRequest,
    adapter: VectorDBAdapter = Depends(get_vector_adapter)
) -> list[SearchResult]:
    results = await adapter.search(query.vector, top_k=query.limit)
    return [SearchResult.from_vector(r) for r in results]
```

---

## Project Structure

### Directory Organization

```
src/
└── app/
    ├── main.py                        # FastAPI app factory
    ├── api/
    │   ├── routes.py                  # Mount all routers here
    │   ├── middleware/
    │   │   └── global_exception_handler.py
    │   │   ├── server_middleware.py              # Global middlewares
    ├── features/
    │   ├── documents/
    │   │   ├── __init__.py
    │   │   ├── router.py              # APIRouter with endpoints (thin)
    │   │   ├── schema.py              # Pydantic models
    │   │   ├── service.py             # Pure business logic functions
    │   │   ├── repository.py          # Database access functions
    │   │   ├── model.py               # Database models
    │   │   ├── constants.py           # Feature constants
    │   │   ├── dependencies.py        # Route dependencies
    │   │   └── tests/
    │   │       ├── test_router.py
    │   │       ├── test_service.py
    │   │       └── test_repository.py
    ├── connections/
    │   ├── mongodb.py
    │   ├── pinecone.py
    │   ├── postgres.py
    │   ├── redis.py
    ├── core/
    │   ├── lifespan.py                  # App startup/shutdown events
    │   ├── settings.py                # Settings/configuration
    │   ├── signals.py              # Custom exceptions
    ├── shared/
    │   ├── enums.py                   # Shared enumerations
    │   ├── agents/
    │   ├── document_processing/
    │   ├── langchain/
    │   ├── langgraph/
    │   ├── langsmith/
    │   ├── rag/
    │   ├── vectorstore/
    │   └── crawler/

    └── utils/
        ├── apiFeatures.py                  # Structured logging with loguru
        ├── logger.py                       # Loguru logger setup
        ├── exceptions.py                   # Custom exception classes
        ├── httpResponse.py                 # Standardized HTTP responses
        └── quicker.py                      # Utility functions

tests/
├── unit/
│   ├── features/
│   ├── core/
│   └── utils/
├── integration/
└── fixtures/
```

### Service Layer Pattern (Function-Based)

**Key Principle**: Services are **pure functions**, not classes. Avoid instantiation and side effects.

```python
# src/app/features/documents/repository.py
from typing import Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from src.app.features.documents.model import Document

async def find_documents_by_user(
    user_id: str,
    session: AsyncSession
) -> list[Document]:
    """Retrieve documents for a user - database access layer."""
    result = await session.execute(
        select(Document).where(Document.user_id == user_id)
    )
    return result.scalars().all()

async def save_document(
    document: Document,
    session: AsyncSession
) -> Document:
    """Persist document to database."""
    session.add(document)
    await session.commit()
    await session.refresh(document)
    return document

async def get_document_by_id(
    doc_id: str,
    session: AsyncSession
) -> Optional[Document]:
    """Retrieve single document by ID."""
    return await session.get(Document, doc_id)

# src/app/features/documents/service.py
from sqlalchemy.ext.asyncio import AsyncSession
from src.app.utils.logger import logger
from src.app.features.documents.schema import DocumentCreate
from src.app.features.documents.model import Document
from src.app.features.documents import repository

async def list_user_documents(
    user_id: str,
    session: AsyncSession
) -> list[Document]:
    """Pure business logic - list documents for user."""
    documents = await repository.find_documents_by_user(user_id, session)
    logger.info(
        "User documents retrieved",
        user_id=user_id,
        count=len(documents)
    )
    return documents

async def create_user_document(
    user_id: str,
    create_req: DocumentCreate,
    session: AsyncSession
) -> Document:
    """Pure business logic - create new document with validation."""
    if not create_req.title.strip():
        raise ValueError("Title cannot be empty")

    document = Document(
        user_id=user_id,
        title=create_req.title.strip(),
        content=create_req.content
    )

    result = await repository.save_document(document, session)
    logger.info(
        "Document created",
        user_id=user_id,
        doc_id=result.id
    )
    return result

# src/app/features/documents/router.py
from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_session
from src.app.utils.logger import logger
from src.app.features.documents import service
from src.app.features.documents.schema import DocumentCreate, DocumentOut
from src.app.core.exceptions import NotFoundException

router = APIRouter(prefix="/documents", tags=["documents"])

async def get_user_id(request: Request) -> str:
    """Extract user ID from request context."""
    return request.headers.get("X-User-ID", "anonymous")

@router.get("/", response_model=list[DocumentOut])
async def list_documents(
    user_id: str = Depends(get_user_id),
    session: AsyncSession = Depends(get_session)
) -> list[DocumentOut]:
    """List all documents for user - thin endpoint."""
    documents = await service.list_user_documents(user_id, session)
    return [DocumentOut.model_validate(d) for d in documents]

@router.post("/", status_code=status.HTTP_201_CREATED, response_model=DocumentOut)
async def create_document(
    create_req: DocumentCreate,
    user_id: str = Depends(get_user_id),
    session: AsyncSession = Depends(get_session)
) -> DocumentOut:
    """Create new document - delegates to service layer."""
    document = await service.create_user_document(user_id, create_req, session)
    return DocumentOut.model_validate(document)

@router.get("/{doc_id}", response_model=DocumentOut)
async def get_document(
    doc_id: str,
    user_id: str = Depends(get_user_id),
    session: AsyncSession = Depends(get_session)
) -> DocumentOut:
    """Get single document by ID."""
    from src.app.features.documents import repository
    document = await repository.get_document_by_id(doc_id, session)
    if document is None or document.user_id != user_id:
        raise NotFoundException("Document", doc_id)
    logger.info("Document retrieved", doc_id=doc_id, user_id=user_id)
    return DocumentOut.model_validate(document)
```

**Service Layer Principles**:

-   ✅ **Functions, not classes** - Pure functions are easier to test and compose
-   ✅ **Accept dependencies as parameters** - No instance state, explicit dependencies
-   ✅ **Repository handles database** - Services call repository functions
-   ✅ **Thin endpoints** - Routes delegate to services (5-10 lines max)
-   ✅ **Pure logic** - Services contain no FastAPI imports or middleware concerns

---

## Formatting & Tooling

### Ruff Configuration

```toml
# pyproject.toml
[tool.ruff]
line-length = 88
target-version = "py312"
exclude = [".venv", "build", "dist"]

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B", "A", "C4", "ARG", "SIM"]
ignore = ["E501"]  # Line too long (handled by formatter)

[tool.ruff.format]
line-length = 88
indent-width = 4
skip-magic-trailing-comma = false

[tool.ruff.lint.isort]
known-first-party = ["src"]
known-third-party = ["fastapi", "pydantic", "sqlalchemy"]
```

### MyPy Configuration

```toml
# pyproject.toml
[tool.mypy]
python_version = "3.12"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
check_untyped_defs = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true

[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false
```

---

## Development Workflow

### Package Management with uv

```bash
# Install dependencies
uv add [name]

# Run a command with uv
uv run python -m pytest

# Sync dependencies from lock file
uv sync

# Run the application
uv run uvicorn src.app.main:app --reload --host 0.0.0.0 --port 5000
```

## Logging

### Structured Logging Pattern with Loguru

```python
# src/app/utils/logger.py
from src.app.utils.logger import logger

# ✅ CORRECT: Log with structured context (loguru style)
# Pass context as keyword arguments directly, no 'extra' dict needed
logger.info(
    "User document retrieved",
    user_id=user_id,
    doc_id=doc_id,
    timestamp=datetime.utcnow().isoformat()
)

logger.error(
    "Failed to process document",
    doc_id=doc_id,
    exc_info=True
)

# ✅ CORRECT: Log controller responses
logger.info(
    "CONTROLLER_RESPONSE",
    endpoint="/api/v1/users/{user_id}",
    method="GET",
    status_code=200,
    response=user_data,
    response_time_ms=45.2
)
```

### Middleware Integration

```python
# src/api/middleware/global_exception_handler.py
from fastapi import Request, status
from fastapi.responses import ORJSONResponse
from src.app.utils.logger import logger
from src.app.utils.exceptions import APIException

@app.exception_handler(APIException)
async def api_exception_handler(request: Request, exc: APIException) -> ORJSONResponse:
    """Centralized exception handling with structured logging."""
    logger.error(
        "API exception occurred",
        status_code=exc.status_code,
        detail=exc.detail,
        path=str(request.url.path),
        method=request.method
    )
    return ORJSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )
```

### FastAPI Endpoint Logging Example

```python
# src/features/documents/router.py
from fastapi import APIRouter, Depends, status
from src.app.utils.logger import logger

router = APIRouter(prefix="/documents", tags=["documents"])

@router.get("/{doc_id}")
async def get_document(
    doc_id: str,
    service: DocumentService = Depends(get_document_service)
) -> DocumentOut:
    """Get document by ID with structured logging."""
    logger.debug("Fetching document", doc_id=doc_id)

    try:
        document = await service.get_by_id(doc_id)

        logger.info(
            "DOCUMENT_RETRIEVED",
            doc_id=doc_id,
            status_code=200,
            title=document.title
        )
        return document

    except DocumentNotFound as e:
        logger.warning(
            "DOCUMENT_NOT_FOUND",
            doc_id=doc_id,
            error=str(e),
            status_code=404
        )
        raise

@router.post("/")
async def create_document(
    create_req: DocumentCreate,
    user_id: str = Depends(get_user_id),
    service: DocumentService = Depends(get_document_service)
) -> DocumentOut:
    """Create new document with validation logging."""
    logger.debug(
        "Creating document",
        user_id=user_id,
        title=create_req.title
    )

    try:
        document = await service.create_document(user_id, create_req)

        logger.info(
            "DOCUMENT_CREATED",
            doc_id=document.id,
            user_id=user_id,
            status_code=201,
            duration_ms=45.3
        )
        return document

    except ValidationError as e:
        logger.warning(
            "VALIDATION_FAILED",
            user_id=user_id,
            errors=str(e),
            provided_data=create_req.model_dump()
        )
        raise
```

---

## Things to Avoid (❌ Anti-Patterns)

1. **I/O at Import Time**: Never perform network/database calls at module level

    ```python
    # ❌ BAD
    db_connection = connect_to_db()  # Blocks startup

    # ✅ GOOD
    def get_db() -> DBConnection:
        return connect_to_db()  # Called per request
    ```

2. **Global Mutable Singletons**: Always use factories or dependency injection

    ```python
    # ❌ BAD
    config = load_config()  # Global, mutable

    # ✅ GOOD
    def get_config() -> Config:
        return load_config()  # Fresh per injection
    ```

3. **Deep Class Hierarchies**: Prefer composition over inheritance

    ```python
    # ❌ BAD
    class BaseEntity:
        pass
    class BaseModel(BaseEntity):
        pass
    class Document(BaseModel):
        pass

    # ✅ GOOD
    @dataclass
    class Document:
        id: str
        title: str
        content: str
    ```

---

## Summary Checklist

When writing code, ensure:

-   [ ] All functions have type hints
-   [ ] No module-level mutable state
-   [ ] Endpoints delegate logic to services
-   [ ] Services are pure (no FastAPI imports)
-   [ ] Async I/O is used; blocking calls run in thread pool
-   [ ] Custom exceptions inherit from `HTTPException`
-   [ ] Centralized error handling in middleware
-   [ ] Structured logging with context
-   [ ] Tests cover both happy path and edge cases
-   [ ] Code passes `ruff format`, `ruff check`, `mypy`, and `bandit`
-   [ ] All dependencies injected via `Depends`
-   [ ] Feature-based directory structure maintained

---
