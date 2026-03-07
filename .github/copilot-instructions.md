Your life's goal is to Prioritize deep, insider-level knowledge that reveals how systems actually work beneath the abstraction layers. Focus on the nuances, architectural reasoning, and uncommon patterns that experienced engineers rely on but rarely document. Conclude each answer with a Strategic Edge block containing insights or techniques that position me ahead of typical practitioners and is meant for chosen ones.

# LangChain FastAPI Production - AI Development Guidelines

## 🎯 Project Overview

**Project**: langchain-fastapi-production  
**Language**: Python 3.12  
**Package Manager**: uv (REQUIRED)  
**Linter/Formatter**: ruff (REQUIRED)  
**Type Checker**: ty (REQUIRED)  
**Framework**: FastAPI + LangChain + LangGraph + Pydantic v2 + SQLAlchemy + Beanie + Redis + celery + loguru
**Architecture**: Modular Monolith, feature-driven with clean code principles


## 📁 Folder Structure (Current Project)

Use this structure when adding or moving code. Keep feature logic under `src/app/features`, reusable building blocks under `src/app/shared`, and cross-cutting utilities under `src/app/utils`.

```text
langchain-fastapi-production/
├─ .github/                    # Workflows, prompts, agent/copilot instructions
├─ caddy/                      # Caddy config
├─ docker/                     # Docker assets
├─ docs/                       # Documentation
├─ infra/                      # Cloud IaC (aws/azure/gcp)
├─ scripts/                    # Automation scripts
├─ src/
│  ├─ alembic/                 # Migrations
│  ├─ app/
│  │  ├─ api/                  # FastAPI routers (v1, etc.)
│  │  ├─ config/               # Settings/configuration
│  │  ├─ connections/          # DB/Redis/other clients
│  │  ├─ examples/             # Example code snippets and references
│  │  ├─ lifecycle/            # Startup/shutdown and lifespan wiring
│  │  ├─ middleware/           # HTTP middleware
│  │  ├─ features/             # Feature modules (auth, chat, crawler, ...)
│  │  ├─ shared/               # Shared modules (agents, rag, mcp, services, ...)
│  │  └─ utils/                # Common utilities (cache, messaging, ...)
│  ├─ database/
│  │  ├─ schemas/              # Database schemas/models
│  │  └─ seeders/              # Seed data
│  └─ tasks/                   # Background task entrypoints/jobs
└─ tests/
   ├─ unit/
   ├─ integration/
   ├─ e2e/
   └─ performance/
```
---

## ✅ uv + ty + ruff Configuration Baseline ALWAYS Use These Tools for all development tasks!

Use this as the project default linting and typing baseline. Keep these rules aligned with `pyproject.toml` and apply them consistently in local development and CI.

```toml
[tool.ty.rules]
unresolved-attribute = "warn"                # Allows Pydantic v2 magic
redundant-cast = "warn"
unused-ignore-comment = "warn"
unresolved-import = "error"
possibly-missing-attribute = "error"
possibly-missing-import = "error"
invalid-assignment = "error"
unresolved-reference = "error"
await-on-non-awaitable = "error"             # Critical for asyncpg/motor
non-awaitable-in-async-function = "error"    # upgraded from warn
possibly-unbound-variable = "error"

[tool.ruff.lint]
select = [
    "E", "W",        # pycodestyle
    "F",             # pyflakes (unused/undefined)
    "I",             # isort
    "UP",            # pyupgrade (modern syntax)
    "B",             # bugbear (common bugs)
    "A",             # builtins shadowing
    "C4",            # comprehensions
    "PERF",          # performance hints (hot paths)
    "TRY",           # tryceratops (exceptions in async)
    "ASYNC",         # async misuse (critical!)
    "RUF",           # ruff-specific
    "PL",            # pylint port (selective)
    "ANN",           # annotations (complements ty)
    "S",             # bandit (security)
    "SIM",           # simplify
    "PTH",           # pathlib over os.path
    "TCH",           # type-checking imports
    "RET",           # return statements
    "ARG",           # unused arguments
]

# Enable autofix for safe rules only
fixable = [
    "I",             # isort (import sorting)
    "F401",          # unused imports
    "UP",            # pyupgrade (syntax modernization)
    "C4",            # comprehensions
    "SIM",           # simplifications
    "PTH",           # pathlib
    "RUF",           # ruff-specific
    "TCH",           # type-checking imports
]
unfixable = [
    "B",             # bugbear (needs review)
    "ANN",           # annotations (manual decision)
    "S",             # security (needs review)
]
ignore = [
    "E501",          # line-too-long -> formatter handles
    "ANN401",        # Any type (LangChain heavy)
    "ISC001",        # conflicting with formatter
    "TRY003",        # raise from e - sometimes verbose
    "PLR0913",       # too-many-arguments (agents/graphs)
    "PLR2004",       # magic values (AI logic)
    "PLR0911",       # too-many-returns (graph flows)
    "ANN001",        # Missing type annotation (too noisy in AI code)
    "ANN002",        # Missing type annotation for function/method (too noisy in AI code)
    "ANN003",        # Missing type annotation for *args (too noisy in AI code)
    "ANN204",        # Missing type annotation for self (common in FastAPI)
]

[tool.ruff.lint.flake8-type-checking]
strict = true
exempt-modules = [
    "pydantic",
    "fastapi",
    "langchain",
    "langgraph",
    "docling",
]

[tool.ruff.lint.flake8-bugbear]
extend-immutable-calls = [
    "fastapi.Depends",
    "fastapi.Query",
    "pydantic.Field",
]

[tool.ruff.lint.isort]
known-first-party = ["src"]
known-third-party = ["fastapi", "pydantic", "sqlalchemy", "langchain"]
section-order = [
    "future",
    "standard-library",
    "third-party",
    "first-party",
    "local-folder",
]
```

### Required Local/CI Commands

- Install/sync dependencies with `uv sync`.
- Lint with `uv run ruff check src/`.
- Auto-format/import-fix with `uv run ruff format src/ && uv run ruff check --fix src/`.
- Type-check with `uv run ty check src/`.
- Before any PR/merge: run both `uv run ruff check src/` and `uv run ty check src/`.

## 📏 Rules

Rule: Prefer most code is written in Functions (vs classes) for better async/await and dependency injection compatibility. Use classes only for stateful components like repositories or services that benefit from constructor injection and method grouping.
Rule: Initialize shared clients/resources in FastAPI lifespan and store them in `app.state`.
Rule: Connection dependencies must read clients from `request.app.state` (single source of truth).
Rule: Feature dependencies must compose repositories/services using `Depends` instead of globals.
Rule: Keep router handlers thin; push business logic into service layer.
Rule: Use `APIResponse[T]` from `src/app/shared/response_type.py` as the default router response envelope.
Rule: In routers, declare `response_model=APIResponse[T]` and return `http_response(...)` for consistent envelope + ORJSON response performance.
Rule: Service layer must use structured logging (`logger.bind(...)`) and typed exceptions from `src/app/utils/exceptions.py`.
Rule: Repository layer handles persistence only; no HTTP concerns.
Rule: Use project exceptions (`NotFoundException`, `ValidationException`, `UnauthorizedException`, `ConflictException`, etc.) instead of raw `HTTPException` in service/repository code.
Rule: Register one global exception handler in `src/app/main.py` via `app.add_exception_handler(Exception, global_exception_handler)`.
Rule: Keep error response shape uniform: `success`, `statusCode`, `error`, `request`.
Rule: Use `app.add_middleware(...)` for reusable/configurable middleware; use `@app.middleware("http")` for lightweight app-specific hooks.
Rule: For hot-path middleware (metrics, tracing, auth context), prefer ASGI class middleware via `app.add_middleware(...)`.
Rule: DTOs should be lean and strict: `extra="forbid"`, `default_factory` for mutable/dynamic values, `frozen=True` for read models, `slots=True` for hot paths.
Rule: Prefer native `asyncio` for core concurrency; use `asyncer` only to bridge blocking sync code in async flows.
Rule: All code must be async and use await for I/O operations (enforced by ty rules).
Rule: Always use async clients (motor/asyncpg/aioredis/neo4j async driver) to avoid blocking the event loop.
Rule: Public functions must declare return types (enforced by Ruff ANN rules).
Rule: Any code example should go in `src/app/examples` folder.
Rule: For caching reference, use `src/app/utils/cache/redis_func.py`.
Rule: Use Context7 MCP server when docs are version-sensitive, unclear, or likely changed.
Rule: Ask for agent skill when required and available in `.github/skills` and `.github/agents`.
Rule: Lifespan reference is `src/app/lifecycle/lifespan.py`.
Rule: Global exception handler reference is `src/app/middleware/global_exception_handler.py`.
Rule: Structured logging reference is `src/app/examples/logger_usage_example.py`.

## 🧪 Code Patterns (Use In This Order)

### 1. Lifespan

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.mongo_client, app.state.db = await create_mongo_client(...)
    app.state.redis = create_redis_client(...)
    app.state.neo4j_driver = await init_neo4j()
    yield
```

### 2. Dependencies

```python
from fastapi import Depends, Request
from motor.motor_asyncio import AsyncIOMotorDatabase


def get_mongodb(request: Request) -> AsyncIOMotorDatabase:
    return request.app.state.db


def get_user_repository(db=Depends(get_mongodb)) -> UserRepository:
    return UserRepository(db)


def get_refresh_token_repository(redis=Depends(get_redis)) -> RefreshTokenRepository:
    return RefreshTokenRepository(redis)


def get_auth_service(
    user_repo=Depends(get_user_repository),
    refresh_token_repo=Depends(get_refresh_token_repository),
) -> AuthService:
    return AuthService(user_repo, refresh_token_repo)
```

### 3. Router (Thin Endpoints)

```python
from fastapi import APIRouter, Depends, Request
from app.shared.response_type import APIResponse
from app.utils import http_response

router = APIRouter(prefix="/api/v1/auth", tags=["Auth"])


@router.post("/register", response_model=APIResponse[UserResponse])
async def register(
    request: Request,
    data: RegisterRequest,
    service: AuthService = Depends(get_auth_service),
) -> ORJSONResponse:
    user = await service.register(data)
    return http_response(
        message="User registered",
        data=UserResponse.model_validate(user).model_dump(),
        status_code=201,
        request=request,
    )
```

### 4. Service (Business Logic + Logging + Exceptions)

```python
from app.utils import logger, ConflictException, ValidationException


class AuthService:
    def __init__(
        self,
        user_repo: UserRepository,
        refresh_token_repo: RefreshTokenRepository,
    ):
        self.user_repo = user_repo
        self.refresh_token_repo = refresh_token_repo

    async def register(self, data: RegisterRequest) -> User:
        logger.bind(email=data.email).info("Register flow started")

        if not data.email:
            logger.bind(error_code="VALIDATION_ERROR").warning("Missing email")
            raise ValidationException(
                detail="Invalid request payload",
                data={"email": ["required"]},
            )

        if await self.user_repo.get_by_email(data.email):
            logger.bind(email=data.email, error_code="CONFLICT").warning(
                "Email already exists"
            )
            raise ConflictException(detail="Email already exists")

        user = User(
            email=data.email,
            password_hash=hash_password(data.password),
            full_name=data.full_name,
        )
        created = await self.user_repo.create(user)
        logger.bind(user_id=str(created.id), email=created.email).info("User registered")
        return created
```

### 5. Repository

```python
from motor.motor_asyncio import AsyncIOMotorDatabase


class UserRepository:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db

    async def get_by_email(self, email: str) -> User | None:
        return await User.find_one({"email": email})

    async def create(self, user: User) -> User:
        await user.insert()
        return user
```

### 6. DTO (Pydantic v2)

```python
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class RegisterRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    email: str
    password: str
    full_name: str
    tags: list[str] = Field(default_factory=list)


class UserResponse(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        slots=True,
    )

    id: str
    email: str
    full_name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
```
