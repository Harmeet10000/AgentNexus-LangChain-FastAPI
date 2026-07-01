# Architecture Rules

## Layering

- Keep router handlers thin; push business logic into the service layer.
- Repository layer handles persistence only; no HTTP concerns.
- Shared clients/resources must be initialized in FastAPI lifespan and stored in `app.state`.
- Connection dependencies must read clients from `connection.app.state` as the single source of truth.
- Feature dependencies must compose repositories and services using `Depends(...)`, not globals.
- Prefer composition over inheritance.
- Prefer functions over classes when no instance state is required.
- Use classes only for stateful components (repositories, services with constructor injection).
- Do not create classes that only group behavior without instance state.
- Do not create classes with only `@staticmethod` helpers; use modules instead.

## Dependency Passing

- Pass dependencies explicitly as function arguments.
- Do not create custom decorators only to inject config, clients, or runtime dependencies.
- When the same group of related dependencies is passed through multiple high-level call layers, prefer a small explicit context object (Pydantic model) instead of repeating long parameter lists.
- Context objects should be narrow and intentional, not god objects.
- Type context behavior fields against `Protocol` interfaces to reduce coupling.
- Use `typing.Protocol` to define the minimum interface a function needs.

## FastAPI Rules

### Routers and endpoints
- Prefer router-level config (`prefix`, `tags`, shared `Depends(...)`) on `APIRouter(...)`.
- Use `APIResponse[T]` from `src/app/shared/response_type.py` as the default response envelope.
- Declare `response_model=APIResponse[T]` and return `http_response(...)`.
- Use `typing.Annotated` for `Path`, `Query`, `Header`, `Cookie`, `Body`, `Depends(...)`.
- Create `Annotated` type aliases for reused dependency patterns.
- Do not use ellipsis (`...`) for required params or Pydantic fields.
- Do not use `Pydantic RootModel` when `Annotated` metadata suffices.

### Dependencies and lifespan
- Use FastAPI dependencies for external resources, cross-endpoint behavior, yield cleanup, sub-dependencies, or request-derived inputs.
- Use `yield` for cleanup; keep default request scope unless cleanup must finish before response.
- Avoid class dependencies; prefer function dependencies returning small objects.
- Lifespan wiring belongs in `src/app/lifecycle/lifespan.py`.

### Streaming
- Prefer generator/async-generator path operations with declared return types.
- For JSON/SSE: return `AsyncIterable[...]` and `yield` typed items.
- For SSE with control: use `response_class=EventSourceResponse` and `yield ServerSentEvent` for explicit control.
- For bytes: declare `response_class=StreamingResponse` and `yield` bytes.
- Use generators instead of materializing large payloads in memory first.

### Middleware and exception handling
- Use `app.add_middleware(...)` for reusable/configurable middleware.
- Use `@app.middleware("http")` only for lightweight app-specific hooks.
- For hot-path (metrics, tracing, auth): prefer ASGI class middleware.
- Register one global exception handler: `app.add_exception_handler(Exception, global_exception_handler)`.
- Uniform error shape: `success`, `statusCode`, `error`, `request`.

## Service and Repository Rules

- Service layer: structured logging, typed project exceptions.
- Repository layer: persistence and data access only.
- Use `NotFoundException`, `ValidationException`, `UnauthorizedException`, `ConflictException` from `src/app/utils/exceptions.py` instead of raw `HTTPException`.
- Use `logger.bind(...)` or structured logging patterns.
- Do not put HTTP response formatting inside repositories.
- Pattern matching: see `RESULT-PATTERN.md` for the decision matrix on when to use `match`/`case` vs `isinstance` vs Result unwrapping.
