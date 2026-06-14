# Python and Typing Rules

## General Style

- Public functions must declare return types.
- Prefer precise types over `Any`.
- Use generics when input and output types are coupled. For Python 3.12+, use PEP 695 syntax: `type Alias[T] = ...`, `class Box[T]: ...`, `def first[T](items: list[T]) -> T`.
- Never use `TypeVar` unless required for Python < 3.12 compat.
- Prefer explicit imports; never use `from module import *`.
- Use generators, generator expressions, and comprehensions when they simplify data flow.
- Do not override dunders (especially `__new__`) in surprising ways.
- Do not use exceptions for normal control flow.
- Keep classes small. Initializers should not perform I/O, DB access, or heavy computation. Delegate to factory functions or class methods (`.from_config()`).

## Properties and Methods

- `@property`: cheap, sync, side-effect-free access only.
- Methods: operations with cost or side effects (I/O, DB, cache, network).
- No persistence or side effects in property setters; use explicit methods.
- No async properties.

## Async and Concurrency

- All I/O code must be async.
- Always use async clients (`motor`, `asyncpg`, `redis.asyncio`, `neo4j`, async LangChain integrations).
- Prefer native `asyncio` for core concurrency.
- Use `asyncer` only to bridge blocking sync code in async flows.
- For bounded fan-out over known small in-memory tasks: `asyncio.gather(...)` + semaphore.
- Do NOT use `gather` + semaphore for bursty/unbounded work.
- For bursty/unbounded: bounded `asyncio.Queue` + worker tasks for backpressure.
- Choose queues when producers can outpace consumers or input size is large.
- Choose direct `gather` only when work is short-lived, bounded, request-scoped.

## Pydantic and DTO Rules

- DTOs should be lean and strict.
- Prefer `extra="forbid"` for request models.
- Use `default_factory` for mutable or dynamic defaults.
- Prefer `frozen=True` for read models.
- Do not manually add field-level `__slots__` to `BaseModel` subclasses.
- Prefer Pydantic dataclasses over `dataclass` and `TypedDict`.
- For collection validation: use `TypeAdapter(list[Model]).validate_python(...)`, not per-item `model_validate` in a loop.
- `ConfigDict(frozen=True)` replaces `@dataclass(frozen=True)`.
- `ConfigDict(arbitrary_types_allowed=True)` for classes holding callables, `Any`, `Runnable`, or non-JSON-serializable types.
- `model_post_init(self, __context: object)` replaces `__post_init__`.
- `Field(default_factory=...)` replaces `field(default_factory=...)`.
