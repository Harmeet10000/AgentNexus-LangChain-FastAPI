---
inclusion: always
---

# FastAPI Patterns & Best Practices

## Response Envelope

Always use `APIResponse[T]` from `src/app/shared/response_type.py`:

```python
from src.app.shared.response_type import APIResponse, http_response

@router.get("/items/{item_id}", response_model=APIResponse[ItemResponse])
async def get_item(item_id: int) -> APIResponse[ItemResponse]:
    item = await service.get_item(item_id)
    return http_response(data=item)
```

Error response shape: `success`, `statusCode`, `error`, `request`

## Dependencies and Annotated Types

Use `typing.Annotated` for all parameters:

```python
from typing import Annotated
from fastapi import Query, Path, Depends

# Good
async def search(
    query: Annotated[str, Query(min_length=1)],
    limit: Annotated[int, Query(gt=0, le=100)] = 10,
) -> APIResponse[list[Item]]:
    pass

# Bad - don't use ellipsis
async def search(query: str = ..., limit: int = ...):
    pass
```

Reuse dependencies with type aliases:

```python
CurrentUser = Annotated[User, Depends(get_current_user)]
AdminUser = Annotated[User, Depends(get_admin_user)]

async def admin_action(user: AdminUser) -> APIResponse[None]:
    pass
```

## Streaming Responses

### JSON/SSE Streaming
```python
async def stream_items() -> AsyncIterable[ItemResponse]:
    for item in items:
        yield item
```

### Server-Sent Events
```python
from sse_starlette.responses import EventSourceResponse

@router.get("/stream")
async def stream() -> EventSourceResponse:
    async def event_generator():
        yield {"data": json.dumps({"message": "hello"})}
    return EventSourceResponse(event_generator())
```

### Byte Streaming
```python
@router.get("/download", response_class=StreamingResponse)
async def download() -> AsyncIterable[bytes]:
    async for chunk in get_file_chunks():
        yield chunk
```

## Middleware and Exception Handling

### Global Exception Handler
Register in `src/app/main.py`:
```python
from src.app.middleware.global_exception_handler import global_exception_handler

app.add_exception_handler(Exception, global_exception_handler)
```

### Middleware Registration
```python
# Reusable/configurable middleware
app.add_middleware(MyMiddleware, option=value)

# Lightweight app-specific hooks only
@app.middleware("http")
async def my_hook(request: Request, call_next):
    response = await call_next(request)
    return response
```

### Hot-Path Middleware
For metrics, tracing, auth context: use ASGI class middleware, not decorators

## Router Configuration

Prefer router-level config:

```python
router = APIRouter(
    prefix="/items",
    tags=["items"],
    dependencies=[Depends(verify_token)],
)

@router.get("/{item_id}")
async def get_item(item_id: int) -> APIResponse[ItemResponse]:
    pass
```

Not at `include_router()` call site.

## Lifespan Management

Use FastAPI dependencies with `yield` for cleanup:

```python
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSession(engine) as session:
        yield session
        # cleanup happens after response sent (default request scope)
```

Use `scope="function"` only when cleanup must finish before response:

```python
async def get_lock() -> AsyncGenerator[Lock, None]:
    lock = Lock()
    await lock.acquire()
    try:
        yield lock
    finally:
        await lock.release()  # must finish before response
```

Lifespan wiring: `src/app/lifecycle/lifespan.py`
