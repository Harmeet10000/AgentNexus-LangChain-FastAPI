# LangChain FastAPI Production - AI Development Guidelines

<Info>
    This document serves as the authoritative guide for AI agents and developers
    working on this project. It covers coding standards, architectural patterns,
    and best practices specific to this codebase.
</Info>

## Project Overview

**Project**: langchain-fastapi-production
**Language**: Python 3.12
**Framework**: FastAPI + LangChain + LangGraph + Pydantic v2 + SQLAlchemy + beanie + Redis + httpx
**Architecture Style**: Modular Monolith, feature-driven with emphasis on clean code principles

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

#### Rules for __init__.py Files

1. **Always include docstrings**: Describe the package purpose
2. **Import all public exports**: Any class, function, or constant that external code should access
3. **Maintain __all__**: Explicitly list all public exports in `__all__`
4. **Handle empty modules**: For modules under development, use placeholder comments:

    ```python
    # src/app/shared/rag/__init__.py
    """RAG (Retrieval-Augmented Generation) utilities."""

    # Import from submodules when they have content
    # from .retriever import ...
    # from .vectorstore import ...

    __all__ = []
    ```

5. **Organize by layer**: Group imports logically

#### Directory Structure for Imports

```

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
use python v3.12+ recommened syntax (PEP 695) with built-in generics and type aliases

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
value: str | None = get_value()
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

### Adapter Pattern
### Repository Pattern

---



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


---

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

