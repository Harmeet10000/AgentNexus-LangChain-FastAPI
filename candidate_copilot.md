# LangChain FastAPI Production - AI Development Guidelines

## 🎯 Project Overview

**Project**: langchain-fastapi-production  
**Language**: Python 3.14  
**Package Manager**: uv (REQUIRED)  
**Linter/Formatter**: ruff (REQUIRED)  
**Type Checker**: ty (REQUIRED)  
**Framework**: FastAPI + LangChain + LangGraph + Pydantic v2 + SQLAlchemy + Beanie + Redis  
**Architecture**: Modular Monolith, feature-driven with clean code principles

---

## 🔧 Tooling Requirements

### ALWAYS Use These Tools

1. **uv** - Fast Python package manager
   ```bash
   # Install dependencies
   uv sync
   uv sync --extra dev
   
   # Add new package
   uv add package-name
   
   # Run commands
   uv run python script.py


Copy

Insert at cursor
markdown
ruff - Fast linter and formatter

# Format code
ruff format .

# Lint code
ruff check .

# Fix auto-fixable issues
ruff check --fix .

Copy

Insert at cursor
bash
ty - Fast type checker

# Type check
ty check src/

Copy

Insert at cursor
bash
📐 Core Architecture Patterns
1. Feature-Driven Structure
src/app/features/{feature_name}/
├── __init__.py          # Export public API
├── model.py             # Beanie/SQLAlchemy models
├── dto.py               # Pydantic request/response schemas
├── router.py            # FastAPI endpoints (thin)
├── service.py           # Business logic
├── repository.py        # Database operations
├── dependency.py        # DI factories
└── constants.py         # Feature constants

Copy

Insert at cursor
2. Dependency Injection Pattern
# ✅ CORRECT: Factory functions for DI
# dependency.py
from fastapi import Depends
from app.connections.mongodb import get_db
from app.connections.redis import get_redis

def get_user_repository(db=Depends(get_db)):
    return UserRepository(db)

def get_refresh_token_repository(redis=Depends(get_redis)):
    return RefreshTokenRepository(redis)

def get_auth_service(
    user_repo=Depends(get_user_repository),
    refresh_token_repo=Depends(get_refresh_token_repository),
):
    return AuthService(user_repo, refresh_token_repo)

# router.py
@router.post("/login")
async def login(
    data: LoginRequest,
    service: AuthService = Depends(get_auth_service),
):
    return await service.login(data.email, data.password)


Copy

Insert at cursor
python
3. Repository Pattern
# ✅ CORRECT: Repository handles all DB operations
# repository.py
from motor.motor_asyncio import AsyncIOMotorDatabase
from app.features.auth.model import User

class UserRepository:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db

    async def get_by_email(self, email: str) -> User | None:
        return await User.find_one({"email": email})

    async def get_by_id(self, user_id: str) -> User | None:
        return await User.get(user_id)

    async def create(self, user: User) -> User:
        await user.insert()
        return user

Copy

Insert at cursor
python
4. Service Layer Pattern
# ✅ CORRECT: Service contains business logic
# service.py
from app.features.auth.repository import UserRepository, RefreshTokenRepository
from app.features.auth.security import hash_password, verify_password
from app.utils.logger import logger

class AuthService:
    def __init__(
        self, 
        user_repo: UserRepository, 
        refresh_token_repo: RefreshTokenRepository
    ):
        self.user_repo = user_repo
        self.refresh_token_repo = refresh_token_repo

    async def register(self, data: RegisterRequest):
        if await self.user_repo.get_by_email(data.email):
            raise ValueError("Email already exists")

        user = User(
            email=data.email,
            password_hash=hash_password(data.password),
            full_name=data.full_name,
        )
        await self.user_repo.create(user)
        logger.info(f"User registered: {data.email}")
        return user


Copy

Insert at cursor
python
5. Router Pattern (Thin Endpoints)
# ✅ CORRECT: Thin routers delegate to services
# router.py
from fastapi import APIRouter, Depends, HTTPException

router = APIRouter(prefix="/api/v1/auth", tags=["Auth"])

@router.post("/register")
async def register(
    data: RegisterRequest,
    service: AuthService = Depends(get_auth_service),
):
    try:
        user = await service.register(data)
        return {
            "id": str(user.id),
            "email": user.email,
            "full_name": user.full_name,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

Copy

Insert at cursor
python
🗄️ Database Patterns
MongoDB with Beanie
# ✅ CORRECT: Beanie document model
from datetime import datetime, timezone
from typing import Annotated
from beanie import Document, Indexed
from pydantic import EmailStr

class User(Document):
    email: Annotated[EmailStr, Indexed(unique=True)]
    password_hash: str
    full_name: str
    created_at: datetime = datetime.now(timezone.utc)
    updated_at: datetime = datetime.now(timezone.utc)

    class Settings:
        name = "users"

Copy

Insert at cursor
python
PostgreSQL with SQLAlchemy
# ✅ CORRECT: Async session pattern
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

engine = create_async_engine(
    get_database_url(),
    echo=False,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


Copy

Insert at cursor
python
Redis Caching
# ✅ CORRECT: Redis client with retry
from redis.asyncio import Redis
from redis.backoff import ExponentialBackoff
from redis.retry import Retry

def create_redis_client(url: str) -> Redis:
    retry_strategy = Retry(
        backoff=ExponentialBackoff(base=0.1, cap=2.0),
        retries=3,
    )

    return Redis.from_url(
        url,
        db=0,
        socket_connect_timeout=120.0,
        socket_timeout=5.0,
        socket_keepalive=True,
        retry=retry_strategy,
        decode_responses=True,
        health_check_interval=30,
    )

Copy

Insert at cursor
python
🔐 Configuration & Settings
Settings Pattern with Pydantic
# ✅ CORRECT: Type-safe settings with lru_cache
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env.development",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    APP_NAME: str = Field(default="LangChain FastAPI Production")
    ENVIRONMENT: str = Field(default="development")
    MONGODB_URI: str = Field(default="mongodb://localhost:27017/db")
    REDIS_URL: str = Field(default="redis://localhost:6379")

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

Copy

Insert at cursor
python
🚀 Application Lifecycle
Lifespan Management
# ✅ CORRECT: Async context manager for startup/shutdown
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logger.info("Application starting")

    # Startup: Initialize connections
    mongo_client, db = await create_mongo_client(
        uri=settings.MONGODB_URI,
        db_name=settings.MONGODB_DB_NAME,
        document_models=[User, Search],
    )
    app.state.mongo_client = mongo_client
    app.state.db = db

    redis = create_redis_client(settings.REDIS_URL)
    app.state.redis = redis

    logger.info("Application ready")
    
    yield
    
    # Shutdown: Close connections
    logger.info("Application shutting down")
    mongo_client.close()
    await redis.close()
    logger.info("Application stopped")


Copy

Insert at cursor
python
🛡️ Middleware Patterns
Middleware Order (CRITICAL)
# ✅ CORRECT: Add middleware in REVERSE order of execution
def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)

    # Last added = First executed
    app.add_middleware(CORSMiddleware, ...)        # 1. First
    app.add_middleware(TrustedHostMiddleware, ...) # 2. Second
    app.add_middleware(GZipMiddleware, ...)        # 3. Third
    app.add_middleware(TimeoutMiddleware, ...)     # 4. Fourth
    app.add_middleware(MetricsMiddleware, ...)     # 5. Fifth

    # Decorator-style middleware (executed after add_middleware)
    @app.middleware("http")
    async def security_headers(request, call_next):
        return await create_security_headers_middleware(request, call_next)

    @app.middleware("http")
    async def correlation_id(request, call_next):
        return await correlation_middleware(request, call_next)

    return app

Copy

Insert at cursor
python
Custom Middleware Pattern
# ✅ CORRECT: Pure ASGI middleware
class MetricsMiddleware:
    def __init__(self, app, project_name: str = "app"):
        self.app = app
        self.project_name = project_name

    async def __call__(self, scope: dict, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start_time = time.perf_counter()
        
        async def send_wrapper(message: dict):
            if message["type"] == "http.response.start":
                duration = time.perf_counter() - start_time
                headers = list(message.get("headers", []))
                headers.append((b"x-process-time", f"{duration:.3f}".encode()))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_wrapper)

Copy

Insert at cursor
python
🎨 Pydantic v2 Patterns
DTO (Data Transfer Objects)
# ✅ CORRECT: Request/Response DTOs
from pydantic import BaseModel, EmailStr, Field, field_validator

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    full_name: str

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class UserResponse(BaseModel):
    id: str
    email: EmailStr
    full_name: str

    model_config = ConfigDict(from_attributes=True)

Copy

Insert at cursor
python
📝 Logging Patterns
Structured Logging with Loguru
# ✅ CORRECT: Structured logging with context
from app.utils.logger import logger

# Log with structured data
logger.info(
    "User registered successfully",
    user_id=user.id,
    email=user.email,
    timestamp=datetime.utcnow().isoformat()
)

# Log errors with context
logger.error(
    "Failed to process document",
    doc_id=doc_id,
    error=str(e),
    exc_info=True
)

# Use contextualize for request-scoped logging
with logger.contextualize(correlation_id=correlation_id, user_id=user_id):
    logger.info("Processing request")

Copy

Insert at cursor
python
⚠️ Error Handling
Custom Exception Pattern
# ✅ CORRECT: Custom API exception
from fastapi import HTTPException

class APIException(HTTPException):
    def __init__(
        self, 
        status_code: int, 
        message: str, 
        data: Any = None, 
        name: str = "APIError"
    ):
        self.name = name
        self.message = message
        self.data = data
        super().__init__(status_code=status_code, detail=message)

Copy

Insert at cursor
python
Global Exception Handler
# ✅ CORRECT: Centralized exception handling
async def global_exception_handler(request: Request, exc: Exception):
    correlation_id = getattr(request.state, "correlation_id", "unknown")
    
    if isinstance(exc, APIException):
        error_obj = {
            "name": exc.name,
            "statusCode": exc.status_code,
            "message": exc.message,
            "correlationId": correlation_id,
        }
        logger.error(f"[{correlation_id}] {exc.name}: {exc.message}")
    else:
        error_obj = {
            "name": "InternalServerError",
            "statusCode": 500,
            "message": "An unexpected error occurred",
            "correlationId": correlation_id,
        }
        logger.error(f"[{correlation_id}] Unexpected error", exc_info=True)

    return ORJSONResponse(
        status_code=error_obj["statusCode"],
        content=error_obj,
    )


Copy

Insert at cursor
python
🧪 Type Hints (Python 3.12+)
# ✅ CORRECT: Modern Python 3.12 type hints
from collections.abc import Awaitable, Callable

# Use built-in generics (no typing.List, typing.Dict)
def process_items(items: list[str]) -> dict[str, int]:
    return {item: len(item) for item in items}

# Union types with |
def get_user(user_id: str) -> User | None:
    pass

# Generic functions
async def fetch_resource[T](
    resource_id: str,
    model: type[T]
) -> T | None:
    pass

# Callable types
def create_handler(
    service: ServiceType
) -> Callable[[Request], Awaitable[Response]]:
    async def handler(request: Request) -> Response:
        pass
    return handler

Copy

Insert at cursor
python
📦 Package Structure & Imports
init.py Pattern
# ✅ CORRECT: Clean package exports
# src/app/connections/__init__.py
"""Database connection dependencies."""

from app.connections.mongodb import create_mongo_client
from app.connections.redis import create_redis_client

__all__ = [
    "create_mongo_client",
    "create_redis_client",
]

Copy

Insert at cursor
python
Import Style
# ✅ CORRECT: Import from package __init__.py
from app.config import Settings, get_settings
from app.connections import create_mongo_client, create_redis_client
from app.utils import logger

# ❌ WRONG: Direct submodule imports
from app.config.settings import get_settings
from app.connections.mongodb import create_mongo_client

Copy

Insert at cursor
python
🚫 Anti-Patterns to Avoid
❌ Module-Level I/O
# ❌ WRONG: I/O at import time
db = connect_to_database()  # Blocks startup!

# ✅ CORRECT: Lazy initialization
def get_db():
    return connect_to_database()

Copy

Insert at cursor
python
❌ Global Mutable State
# ❌ WRONG: Global mutable singleton
config = load_config()

# ✅ CORRECT: Factory with caching
@lru_cache(maxsize=1)
def get_config():
    return load_config()

Copy

Insert at cursor
python
❌ Fat Endpoints
# ❌ WRONG: Business logic in router
@router.post("/users")
async def create_user(data: UserCreate):
    # Validation logic
    # Database operations
    # Email sending
    # Logging
    # 50+ lines of code

# ✅ CORRECT: Thin endpoint
@router.post("/users")
async def create_user(
    data: UserCreate,
    service: UserService = Depends(get_user_service),
):
    return await service.create_user(data)

Copy

Insert at cursor
python
🎯 Code Quality Checklist
Before committing code, ensure:

 uv used for dependency management
 ruff format applied to all files
 ruff check --fix passed with no errors
 ty check passed with no type errors
 All functions have type hints
 All async I/O uses await
 No blocking operations in async functions
 Endpoints are thin (< 15 lines)
 Business logic in service layer
 Database operations in repository layer
 Proper error handling with custom exceptions
 Structured logging with context
 init.py exports public API
 No module-level mutable state
 Dependency injection used throughout
📚 Quick Reference
Command Cheat Sheet
# Development
uv sync --extra dev
uv run uvicorn src.app.main:app --reload

# Code Quality
ruff format .
ruff check --fix .
ty check src/

# Testing
uv run pytest
uv run pytest --cov=src

# Dependencies
uv add package-name
uv add --dev package-name
uv lock

Copy

Insert at cursor
bash
File Naming Conventions
model.py - Database models (Beanie/SQLAlchemy)

dto.py - Pydantic request/response schemas

router.py - FastAPI endpoints

service.py - Business logic

repository.py - Database operations

dependency.py - DI factories

constants.py - Feature constants

Remember: Always use uv + ruff + ty for all development tasks!
