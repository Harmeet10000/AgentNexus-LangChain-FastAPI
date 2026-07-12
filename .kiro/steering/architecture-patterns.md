---
inclusion: always
---

# Architecture Patterns & Layering

## Core Principles

This is a modular monolith with feature-driven, async-first architecture. Every feature is self-contained with complete layer stack: Router → Service → Repository → Model.

## Layering Rules

### Router Layer (FastAPI)
- Keep handlers thin; push business logic to service layer
- Use `APIResponse[T]` from `src/app/shared/response_type.py` as default envelope
- Declare `response_model=APIResponse[T]` and return `http_response(...)`
- Prefer `typing.Annotated` for parameters: `Annotated[int, Query()]` not `...`
- Prefer router-level config (`prefix`, `tags`, dependencies) on `APIRouter(...)`
- Never use ellipsis (`...`) for required parameters or Pydantic fields

### Service Layer
- Contains business logic and LangChain/LangGraph integration
- Use structured logging with `logger.bind(...)`
- Raise typed exceptions: `NotFoundException`, `ValidationException`, `UnauthorizedException`, `ConflictException`
- No HTTP concerns; no response formatting

### Repository Layer
- Persistence and data access only
- No HTTP concerns
- No business logic
- Use async clients: `motor`, `asyncpg`, `redis.asyncio`, `neo4j`

### Dependency Passing
- Pass dependencies explicitly as function arguments
- No custom decorators for injection
- Use small, narrow context objects (dataclasses) at orchestration boundaries
- Type context fields against `Protocol` interfaces to reduce coupling
- Low-level helpers receive only specific arguments they need

## Composition Over Inheritance

- Prefer composition: combine small collaborators, protocols, helper functions
- Use classes only for stateful components (repositories, services)
- Prefer functions over classes when no instance state required
- Never create classes that only group behavior without instance state
- Never create classes with only `@staticmethod` helpers; use modules instead

## Shared Resources

- Initialize all shared clients in FastAPI lifespan
- Store in `app.state` as single source of truth
- Connection dependencies read from `request.app.state`
- Lifespan wiring: `src/app/lifecycle/lifespan.py`

## Feature Structure

```
src/app/features/[feature-name]/
├── __init__.py
├── router.py              # FastAPI routes
├── service.py             # Business logic
├── repository.py          # Data access
├── schemas.py             # Pydantic models
├── models.py              # SQLAlchemy models
├── dependencies.py        # FastAPI dependencies
├── exceptions.py          # Feature exceptions
└── constants.py           # Feature constants
```

## Shared Subsystems

```
src/app/shared/
├── agents/                # Agent runtime building blocks
│   ├── memory/            # Memory managers/integrations
│   ├── orchestration/     # Routing/supervision logic
│   └── tools/             # Tool implementations
├── langchain_layer/       # LangChain adapters/components
├── langgraph_layer/       # LangGraph graphs/nodes/state
├── mcp/                   # MCP integrations
├── rag/                   # Retrieval and knowledge layer
├── vectorstore/           # Vector store integrations
└── services/              # Shared service modules
```
