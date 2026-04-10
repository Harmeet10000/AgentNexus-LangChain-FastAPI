# Backend Design Patterns Reference

This sheet turns each pattern into a practical backend reference. For every pattern, the goal is the same: identify the problem, name the solution, show how to apply it, explain the benefits, and keep a reality check so the pattern does not turn into ceremony.

## 1. Adapter Pattern
**Problem:** You need to integrate a third-party service or legacy module, but its interface does not match what your application expects.

**Solution:** Wrap the external dependency in an adapter that translates its API into the interface your codebase already uses.

**How to do it:** Define the interface your app wants, create an adapter class or function around the vendor client, and let business logic depend only on the adapter contract.

**Benefits:** Makes provider swaps easier, isolates vendor-specific quirks, and reduces ripple effects across the codebase.

**Reality check:** If the mismatch is tiny and used in one place, a thin function may be enough. Do not invent a heavy abstraction for a single direct call.

## 2. Facade Pattern
**Problem:** A subsystem is too complex for callers to use directly because it spans multiple services, libraries, or internal steps.

**Solution:** Create a facade that exposes one simple entry point while hiding the underlying workflow.

**How to do it:** Build a service layer such as `PaymentService` or `NotificationService` that coordinates lower-level clients and exposes a small, stable API.

**Benefits:** Reduces coupling, simplifies caller code, and gives you one place to enforce orchestration rules.

**Reality check:** A facade can become a dumping ground. Keep its responsibilities focused instead of turning it into a god service.

## 3. DTO Pattern
**Problem:** Internal models often contain fields, relationships, or naming conventions that should not be exposed directly to API consumers.

**Solution:** Use Data Transfer Objects to define explicit request and response shapes.

**How to do it:** Map domain or ORM objects to dedicated schemas, usually with Pydantic models or typed serializers, and return those instead of raw database entities.

**Benefits:** Prevents accidental data leaks, stabilizes API contracts, and lets internal models evolve independently.

**Reality check:** DTOs add mapping code. If your internal model and API contract are genuinely identical and stable, an extra layer may be wasteful.

## 4. Proxy Pattern
**Problem:** You need controlled access to an object for concerns like lazy loading, caching, logging, authorization, or rate limiting.

**Solution:** Place a proxy in front of the real object so calls flow through a controlled layer first.

**How to do it:** Define the same interface as the target object, then delegate calls while injecting the extra behavior before or after execution.

**Benefits:** Keeps cross-cutting access rules out of core logic and makes expensive resources easier to manage.

**Reality check:** In many backend systems, middleware or decorators already solve this more simply. Use a proxy when object-level control is actually needed.

## 5. Decorator Pattern
**Problem:** You want to add behavior around a function or class without changing its core implementation.

**Solution:** Wrap the target with a decorator that adds pre- or post-processing.

**How to do it:** In Python, implement function decorators for concerns like auth, retries, metrics, tracing, or rate limits, and apply them where needed.

**Benefits:** Composes well, keeps business code clean, and makes cross-cutting concerns reusable.

**Reality check:** Stacked decorators can hide execution flow. If debugging becomes hard, the abstraction is too dense.

## 6. Dependency Injection Pattern
**Problem:** Hardcoded dependencies make code rigid, hard to test, and difficult to replace across environments.

**Solution:** Pass dependencies into a class or function instead of creating them internally.

**How to do it:** Define constructor parameters or provider functions for repositories, clients, config, and services, then wire them together at the application boundary.

**Benefits:** Improves testability, supports environment-specific implementations, and makes dependency graphs explicit.

**Reality check:** Lightweight manual DI is often enough. Do not bring in a full DI container unless the project complexity justifies it.

## 7. Factory Pattern
**Problem:** Object creation logic is branching, repetitive, or dependent on configuration or runtime context.

**Solution:** Centralize creation inside a factory instead of scattering `if/else` construction logic across the codebase.

**How to do it:** Create a function or class that takes a type, config, or strategy key and returns the correct implementation.

**Benefits:** Reduces duplication, keeps construction logic consistent, and makes new variants easier to add.

**Reality check:** If there are only two straightforward cases, a plain conditional near the call site may be clearer than a separate factory.

## 8. Singleton Pattern
**Problem:** Some resources should have one shared application-level instance, such as config loaders or connection pools.

**Solution:** Ensure only one instance is created and reused.

**How to do it:** Create the shared object once at startup, then inject or reference it from the rest of the app rather than re-instantiating it.

**Benefits:** Prevents duplicate heavy initialization and creates a consistent shared resource lifecycle.

**Reality check:** Singletons often hide global state and make tests brittle. Prefer explicit ownership and injection over magical access.

## 9. Builder Pattern
**Problem:** Creating a complex object requires too many optional parameters, conditional steps, or intermediate validation.

**Solution:** Use a builder to assemble the object step by step.

**How to do it:** Create a builder with chained methods or staged construction, validate the required inputs, and produce the final object through a `build()` step.

**Benefits:** Improves readability for complex construction and makes optional combinations easier to manage safely.

**Reality check:** For simple Python objects, keyword arguments or helper functions are usually cleaner than a full builder.

## 10. Repository Pattern
**Problem:** Business logic becomes tightly coupled to SQL queries, ORM details, or storage-specific code.

**Solution:** Put persistence access behind repositories that expose domain-oriented methods.

**How to do it:** Create repository interfaces and implementations such as `UserRepository.get_by_id()` or `OrderRepository.save()`, then keep query details inside them.

**Benefits:** Improves separation of concerns, makes tests easier, and keeps service code focused on use cases instead of persistence mechanics.

**Reality check:** Repositories can become pointless wrappers over the ORM if they add no real abstraction. Only introduce them where they clarify boundaries.

## 11. Observer Pattern
**Problem:** Multiple components need to react when one component changes state, but direct coupling would become messy.

**Solution:** Use an observer or publish-subscribe model so subscribers are notified when an event occurs.

**How to do it:** Emit events from the source and register listeners, handlers, or subscribers that consume those events asynchronously or synchronously.

**Benefits:** Decouples producers from consumers and makes it easier to extend behavior without editing the source component.

**Reality check:** Event-driven flows are harder to trace and debug. Use them when decoupling matters, not as a substitute for ordinary function calls.

## 12. Strategy Pattern
**Problem:** You have multiple interchangeable ways to perform the same task, and large conditionals are spreading through the code.

**Solution:** Encapsulate each algorithm in its own strategy and select one at runtime.

**How to do it:** Define a common interface, implement one class or function per variant, and inject the selected strategy based on context or configuration.

**Benefits:** Keeps branching out of core logic and makes new behaviors easier to add safely.

**Reality check:** If the variation is tiny and unlikely to grow, a simple conditional is more honest than several strategy classes.

## 13. Command Pattern
**Problem:** You want to represent work as a discrete unit that can be queued, retried, logged, scheduled, or audited.

**Solution:** Wrap the action and its inputs into a command object or message.

**How to do it:** Define a command structure with all required data, then pass it to a handler or worker that knows how to execute it.

**Benefits:** Fits background jobs well, supports retry semantics, and makes workflows easier to serialize or trace.

**Reality check:** Not every function call needs to become a command. Use it when the work needs lifecycle management beyond immediate execution.

## 14. Template Method Pattern
**Problem:** Several workflows share the same overall structure but differ in a few steps.

**Solution:** Define the algorithm skeleton once and allow specific steps to vary.

**How to do it:** Put the common flow in a base class or shared function and let subclasses or injected hooks override the variable parts.

**Benefits:** Removes duplication in repeated workflows and preserves a consistent high-level process.

**Reality check:** In Python, composition is often cleaner than inheritance. Prefer hooks or composed functions unless inheritance truly simplifies things.

## 15. Chain of Responsibility Pattern
**Problem:** A request should pass through multiple handlers, and each handler may process, modify, reject, or forward it.

**Solution:** Arrange handlers in a chain where each one decides whether to continue.

**How to do it:** Build middleware or handler pipelines for validation, authentication, logging, transformation, or error handling.

**Benefits:** Keeps each step focused and supports flexible request processing pipelines.

**Reality check:** Long chains can obscure where behavior actually happens. Keep the order explicit and the number of handlers reasonable.

## 16. Circuit Breaker Pattern
**Problem:** Repeated calls to a failing downstream service can consume resources and trigger cascading failures.

**Solution:** Stop calling the dependency temporarily once failures cross a threshold, then retry later in a controlled way.

**How to do it:** Track error rates and timeouts, open the breaker when the threshold is exceeded, fail fast while open, and move to half-open after a cooldown period.

**Benefits:** Protects your service, limits blast radius, and gives unstable dependencies time to recover.

**Reality check:** A breaker does not fix a bad dependency. It only buys safety. You still need fallbacks, alerts, and operational visibility.

## 17. Bulkhead Pattern
**Problem:** One overloaded dependency or workload can consume all shared resources and drag unrelated parts of the system down with it.

**Solution:** Isolate critical resources into separate pools so failure or saturation in one area does not take down everything else.

**How to do it:** Split worker pools, connection pools, queues, thread limits, or async semaphores by workload or dependency, then enforce clear capacity boundaries for each segment.

**Benefits:** Reduces blast radius, protects critical paths, and keeps one noisy subsystem from starving the rest of the application.

**Reality check:** Bulkheads do not create more capacity. They only partition it. If the limits are set badly, you can waste resources or throttle healthy traffic unnecessarily.

## 18. Saga Pattern
**Problem:** A business transaction spans multiple services, but you cannot use a single ACID transaction across them.

**Solution:** Break the workflow into local transactions coordinated through forward steps and compensating actions.

**How to do it:** Define the sequence of service actions, record progress, and implement compensations for each step that must be reversed on failure.

**Benefits:** Enables distributed workflows while preserving eventual consistency across services.

**Reality check:** Sagas are operationally expensive and hard to reason about. If a modular monolith can solve the problem, that is often the better system.

## 19. Return Result Pattern
**Problem:** Exception-heavy flows can make expected failures noisy, hard to reason about, and expensive to handle at scale.

**Solution:** Return explicit success or failure results instead of using exceptions for normal control flow.

**How to do it:** Define a consistent result shape such as `Ok/Err`, `Success/Failure`, or typed response objects, then make callers handle both paths deliberately.

**Benefits:** Makes expected failure states visible in the function contract and can simplify validation and domain-level error handling.

**Reality check:** Do not force this everywhere. For truly exceptional failures, exceptions are still the right tool. Mixing both styles without discipline creates confusion.

## 20. Unit of Work Pattern
**Problem:** A use case touches several entities or repositories, and you need them committed as one business transaction.

**Solution:** Track all changes inside one unit of work and commit or roll back them together.

**How to do it:** Open a transactional session, perform related repository operations inside it, then commit once the full use case succeeds.

**Benefits:** Preserves consistency within a transaction boundary and gives a clean model for rollback behavior.

**Reality check:** If each request only performs one simple write, an explicit unit-of-work abstraction may just duplicate what your ORM session already does.

## 21. Outbox Pattern
**Problem:** You need to update the database and publish an event, but a failure between those two operations can create inconsistency.

**Solution:** Write the business data and the outbound event to the same database transaction, then publish the event asynchronously from the outbox table.

**How to do it:** Persist the domain change and outbox record together, run a background dispatcher that reads unpublished outbox rows, sends them to the broker, and marks them processed.

**Benefits:** Prevents lost messages, avoids dual-write inconsistency, and is one of the safest ways to integrate databases with message brokers.

**Reality check:** The outbox pattern adds operational overhead, retry logic, idempotency requirements, and cleanup needs. It is worth it only when consistency really matters.

## Final Rule
The main trap is not failing to use patterns. It is using them too early.

Use the rule of three:
1. Write the simplest thing that works.
2. Repeat it once if needed.
3. Refactor into a pattern when repetition or complexity becomes real.

In backend systems, the highest-value patterns are usually the ones that solve boundary and failure problems: DTO, Repository, Unit of Work, Circuit Breaker, Bulkhead, Saga, and Outbox. The rest matter too, but only when the code has actually earned the abstraction.

# Design Patterns For This Repo

This is not a general software patterns catalog. It is a project opinion.

The goal of this codebase is not to maximize architectural ceremony. The goal is to keep a modular monolith easy to change, easy to test, and easy to reason about under async, RAG, agent, and API-heavy workloads.

Prefer small functions, explicit inputs, explicit outputs, and thin boundaries around side effects. When a pattern adds vocabulary without removing complexity, do not use it.

## Patterns We Prefer

### 1. Pure Core, Impure Shell

Keep business rules and transforms pure. Keep I/O at the edges.

Pure functions should:

- accept plain inputs
- return plain outputs
- avoid hidden reads and writes
- be easy to unit test without mocks

The shell should:

- load state
- call external services
- persist results
- log and trace
- call pure functions to do the real work

This is the highest-value pattern in the repo because it keeps RAG pipelines, agent transforms, and response shaping understandable.

```python
def select_chunks(chunks: list[Chunk], limit: int) -> list[Chunk]:
    return sorted(chunks, key=lambda chunk: chunk.score, reverse=True)[:limit]


async def build_context(query: str, retriever: Retriever) -> list[Chunk]:
    chunks = await retriever.search(query)
    return select_chunks(chunks, limit=8)
```

### 2. Result Values For Expected Failures

Do not use exceptions for normal control flow.

When failure is expected and part of the contract, return it explicitly. This is especially useful for validation, tool execution, parsing, and domain-level decision points.

Typical shapes:

- `Ok[T] | Err[E]`
- `Success[T] | Failure[E]`
- project-specific typed result models

Use exceptions for truly exceptional failures: broken infrastructure, invalid invariants, programmer errors, and unexpected library behavior.

```python
class ParseSuccess[T](BaseModel):
    ok: Literal[True] = True
    value: T


class ParseFailure(BaseModel):
    ok: Literal[False] = False
    reason: str


type ParseResult[T] = ParseSuccess[T] | ParseFailure
```

### 3. Small Composable Functions

Prefer functions over classes when no instance state is required.

A function is usually the best unit of reuse for:

- normalization
- validation
- transformation
- ranking
- filtering
- routing decisions
- prompt construction
- response shaping

Use classes only when there is real owned state or a lifecycle concern, such as a repository, connection wrapper, or runtime client.

Bad signs:

- classes that only group methods
- classes with only `@staticmethod`
- services that mostly forward calls

### 4. Context Objects Instead Of DI Plumbing

When the same related dependencies are passed through multiple orchestration layers, group them into a narrow context object.

A good context object:

- is explicit
- is small
- is scoped to one use case or workflow
- contains related runtime services and metadata

A bad context object:

- becomes a god object
- is passed everywhere
- mixes unrelated concerns

Good examples:

- `AgentContext`
- `IngestionContext`
- `SearchContext`

This pattern improves signatures without introducing hidden injection.

### 5. Function-Based Strategy Dispatch

When behavior varies by mode, provider, or content type, prefer function dispatch over large classes.

Use:

- dicts of handlers
- `match` on a small enum
- registries of callables
- protocol-based function arguments

This works well for:

- provider-specific adapters
- chunking strategies
- embedding model selection
- retrieval mode selection
- output formatting modes

```python
type Chunker = Callable[[str], list[str]]


def split_markdown(text: str) -> list[str]:
    ...


def split_plaintext(text: str) -> list[str]:
    ...


CHUNKERS: dict[str, Chunker] = {
    "markdown": split_markdown,
    "plain": split_plaintext,
}
```

### 6. Pipeline Stages

For multi-step flows, use explicit stage-by-stage pipelines.

Good fit:

- ingestion
- extraction
- enrichment
- ranking
- agent pre-processing
- post-processing

Each stage should have one clear job and a clear contract between stages.

Example:

1. load input
2. normalize
3. extract structure
4. enrich with metadata
5. persist
6. emit response

This is easier to debug than a large service method that does everything.

### 7. Boundary Adapters

External systems should be wrapped at the boundary, not leak through the whole codebase.

Use adapters around:

- vendor SDKs
- HTTP APIs
- LLM providers
- vector stores
- graph stores
- indexing SDKs

The rest of the code should depend on the narrow behavior the app needs, not the full vendor surface area.

This keeps swaps, mocks, retries, and error normalization local.


### 9. Idempotent Command Handlers

When work may be retried, make the handler idempotent.

This matters for:

- background jobs
- webhook handlers
- ingestion retries
- tool execution with retry policies
- outbox consumers

The useful pattern is not "Command Pattern" as ceremony. The useful part is:

- one explicit input
- one explicit handler
- clear side effects
- safe re-execution

### 10. Data-Last Helper Functions

Where it improves readability, write helper functions so the main data flows last.

This makes partial application and composition easier.

```python
def take(limit: int, items: Sequence[T]) -> Sequence[T]:
    return items[:limit]


def min_score(threshold: float, items: Sequence[ScoredChunk]) -> list[ScoredChunk]:
    return [item for item in items if item.score >= threshold]
```

This is optional, not dogma. Use it where it makes pipelines read better.

### 11. Explicit Effect Wrappers

Side effects should be obvious in code.

Wrap operations with external impact behind named functions such as:

- `fetch_embeddings(...)`
- `persist_chunks(...)`
- `send_webhook(...)`
- `load_agent_state(...)`
- `publish_outbox_event(...)`

Do not hide network or persistence work behind innocent-looking properties or ambiguous helpers.

### 12. Normalize Once At The Boundary

Normalize data as early as possible, once.

Examples:

- convert provider responses into local typed models
- normalize casing and field names at ingress
- convert vendor exceptions into project exceptions at the adapter boundary
- normalize message history before entering agent logic

This prevents every downstream function from re-defending itself against the same shape problems.

## Pattern Selection Rules

Use a pattern only when it reduces cognitive load.

Ask:

1. Does this remove repeated logic?
2. Does this clarify a boundary?
3. Does this make async or failure behavior easier to reason about?
4. Does this improve testability without increasing ceremony?

If the answer is no, do not introduce the pattern.

## Default Design Bias

When in doubt, prefer:

1. plain functions
2. explicit arguments
3. narrow context objects
4. typed boundaries
5. pure transforms
6. idempotent side-effect handlers

When in doubt, avoid:

1. mirror-model mapping layers
2. service-class sprawl
3. hidden injection
4. distributed workflow machinery
5. abstractions added before the duplication is real

The right abstraction in this repo is usually the one that makes the next edit smaller.


 the traditional, inheritance-heavy State design pattern in Python, proposing instead a cleaner, data-driven approach using modern language features.

Core Concepts & Evolution:
The Traditional Approach (4:11 - 4:57): The video begins by demonstrating the classic OO version using multiple small classes and inheritance. While it keeps state behavior separated, it leads to code duplication and scattered logic.
The State Machine as Data (4:57 - 5:21): Arjan argues that a state machine is essentially just a lookup table (mapping current state + event to a next state + action). This realization allows for a more functional, data-driven design.
Building a Generic Engine (5:21 - 17:35): The video introduces a generic StateMachine class using Python's dataclasses and generics. It uses a dictionary to store transitions, eliminating the need for rigid class hierarchies.
Declarative Transitions (17:35 - 20:37): To make the code even cleaner, Arjan introduces a decorator (@transition). This allows developers to define state transitions directly above the functions that trigger them, making the logic much easier to read.
Advanced Features (20:37 - 24:05): The engine is refined to support multiple 'from' states (e.g., using an iterable in the decorator), ensuring the machine is both flexible and powerful.
Key Takeaways:
Maintainability: By using a centralized engine, the Payment class becomes simple and "boring" (24:05), delegating state-transition logic to the machine.
Open-Closed Principle: You can add new transitions simply by adding new functions and decorators without modifying existing logic (23:36).
When to use: While the data-driven approach is generally superior for its clarity and reusability, Arjan notes that the classic OO pattern might still be preferred if states require extremely complex, data-heavy internal behavior (25:30).
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable

type Action[C] = Callable[[C], None]


class InvalidTransition(Exception):
    pass


@dataclass
class StateMachine[S: Enum, E: Enum, C]:
    transitions: dict[tuple[S, E], tuple[S, Action[C]]] = field(
        default_factory=dict[tuple[S, E], tuple[S, Action[C]]]
    )

    def add_transition(
        self, from_state: S, event: E, to_state: S, func: Action[C]
    ) -> None:
        self.transitions[(from_state, event)] = (to_state, func)

    def next_transition(self, state: S, event: E) -> tuple[S, Action[C]]:
        try:
            return self.transitions[(state, event)]
        except KeyError as e:
            raise InvalidTransition(f"Cannot {event.name} when {state.name}") from e

    def handle(self, ctx: C, state: S, event: E) -> S:
        next_state, action = self.next_transition(state, event)
        action(ctx)
        return next_state

    def transition(self, from_state: S | Iterable[S], event: E, to_state: S):
        if not isinstance(from_state, Iterable):
            from_state = (from_state,)

        def decorator(func: Action[C]) -> Action[C]:
            for s in from_state:
                self.add_transition(s, event, to_state, func)
            return func

        return decorator
        
from dataclasses import dataclass, field
from enum import Enum, auto

from sm import StateMachine


class PayState(Enum):
    NEW = auto()
    AUTHORIZED = auto()
    CAPTURED = auto()
    FAILED = auto()
    REFUNDED = auto()


class PayEvent(Enum):
    AUTHORIZE = auto()
    CAPTURE = auto()
    FAIL = auto()
    REFUND = auto()


@dataclass
class PaymentCtx:
    payment_id: str
    audit: list[str] = field(default_factory=list[str])


# Create an instance: this is "the machine"
pay_sm: StateMachine[PayState, PayEvent, PaymentCtx] = StateMachine()


@pay_sm.transition(PayState.NEW, PayEvent.AUTHORIZE, PayState.AUTHORIZED)
def authorize(ctx: PaymentCtx) -> None:
    ctx.audit.append(f"{ctx.payment_id}: authorized")


@pay_sm.transition((PayState.NEW, PayState.AUTHORIZED), PayEvent.FAIL, PayState.FAILED)
def fail(ctx: PaymentCtx) -> None:
    ctx.audit.append(f"{ctx.payment_id}: failed")


@pay_sm.transition(PayState.AUTHORIZED, PayEvent.CAPTURE, PayState.CAPTURED)
def capture(ctx: PaymentCtx) -> None:
    ctx.audit.append(f"{ctx.payment_id}: captured")


@pay_sm.transition(
    (PayState.AUTHORIZED, PayState.CAPTURED), PayEvent.REFUND, PayState.REFUNDED
)
def refund(ctx: PaymentCtx) -> None:
    ctx.audit.append(f"{ctx.payment_id}: refunded")


@dataclass
class Payment:
    ctx: PaymentCtx
    state: PayState = PayState.NEW

    def handle(self, event: PayEvent) -> None:
        self.state = pay_sm.handle(self.ctx, self.state, event)


def main():
    p = Payment(ctx=PaymentCtx("p1"))

    p.handle(PayEvent.AUTHORIZE)
    p.handle(PayEvent.CAPTURE)
    p.handle(PayEvent.REFUND)

    print("state:", p.state)
    print("audit:", p.ctx.audit)

    # Uncomment to see an invalid transition:
    # p2 = Payment(ctx=PaymentCtx("p2", []))
    # p2.handle(PayEvent.CAPTURE)


if __name__ == "__main__":
    main()
