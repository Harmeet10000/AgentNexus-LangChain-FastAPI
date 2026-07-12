---
inclusion: always
---

# Python & Typing Standards

## Type Hints

- Public functions must declare return types
- Prefer precise types over `Any`
- Use generics when input/output types are coupled (envelopes, containers, repositories)
- Python 3.12+: prefer PEP 695 syntax: `type Alias[T] = ...`, `class Box[T]: ...`, `def first[T](items: list[T]) -> T`
- Keep generic abstractions pragmatic; don't introduce `TypeVar` without real value

## Imports

- Explicit imports only; never `from module import *`
- Keep dependencies traceable and namespaces predictable
- Import order: stdlib, third-party, first-party
- First-party package: `src`
- Known third-party: `fastapi`, `pydantic`, `sqlalchemy`, `langchain`

## Collection Processing

- Use generator functions, generator expressions, comprehensions
- Simplify data flow without reducing readability
- Prefer Python's strengths for iteration

## Control Flow

- Never use exceptions for normal control flow
- Prefer explicit condition checks and branch logic
- Reserve exceptions for exceptional cases

## Dunder Methods

- Never override `__new__` to return unrelated types or hide factory logic
- Use clear factory functions or explicit mappings instead

## Pydantic v2 Models

### Request Models
```python
class CreateItemRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str | None = None
```

### Response Models
```python
class ItemResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: int
    name: str
    created_at: datetime
```

### Defaults
- Use `default_factory` for mutable or dynamic defaults
- Never use `...` (ellipsis) for required fields
- Use `Field(gt=0)` for validation

### Performance
For large collections, use `TypeAdapter` instead of repeated `model_validate()`:

```python
# Bad
users = [User.model_validate(u) for u in raw_users]

# Good
users = TypeAdapter(list[User]).validate_python(raw_users)
```

### Slots Optimization
- Don't manually add `__slots__` to `BaseModel` subclasses as default
- For hot-path, short-lived in-memory containers: use slotted `dataclass` instead

## Protocols for Loose Coupling

Use `typing.Protocol` to define behavior contracts:

```python
from typing import Protocol

class DocumentStore(Protocol):
    async def get(self, key: str) -> str: ...
    async def set(self, key: str, value: str) -> None: ...

async def process(store: DocumentStore) -> None:
    # Works with any implementation of DocumentStore
    await store.set("key", "value")
```

Benefits:
- Reduce coupling to concrete classes
- Improve testability with mock implementations
- Enable duck typing with type safety
