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
