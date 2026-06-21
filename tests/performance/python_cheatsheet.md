
# copilot instructions
## Deployment and Runtime Performance Rules

- Treat Gunicorn `--preload` as a deployment optimization for multi-worker Linux containers: preload mostly immutable app state in the master process before forking so workers can share memory through Copy-on-Write.
- Do not rely on Gunicorn `--preload` for mutable caches, per-worker state, or startup code with side effects that should run independently in each worker.
- Treat `jemalloc` as an infrastructure/runtime optimization to mention during memory tuning, especially for multi-worker API containers. It is not a Python code pattern and should not drive application design.


# Memory Optimization Cheatsheet

This file is a quick reference for memory-focused optimization decisions in this project.


## 3. Use Gunicorn `--preload` only when deployment matches

- `--preload` loads the app in the master process before worker fork.
- This can reduce memory in multi-worker Linux deployments because workers share unchanged memory pages through Copy-on-Write.
- Helps most when startup loads:
  - large modules
  - model metadata
  - mostly immutable lookup tables
  - expensive app wiring
- Be careful:
  - do not depend on it for mutable globals
  - do not preload code with side effects that should run per worker
  - open network connections in lifespan/per-worker startup, not at import time
- This matters only if the app is actually deployed behind Gunicorn. It does not apply to plain `uvicorn.run(...)` directly.

## 4. `jemalloc` is an infra optimization

- `jemalloc` is a memory allocator, not a Python coding pattern.
- It can reduce memory fragmentation and often lowers RSS in multi-worker API containers.
- It is worth testing when:
  - idle memory is too high
  - RSS keeps growing more than expected
  - multi-worker deployments duplicate allocator overhead badly
- Do not encode `jemalloc` assumptions into application code.
- Benchmark it in the actual container/runtime environment before treating it as a win.

## 5. Avoid accidental response and object duplication

- Do not convert large iterables into `list(...)` unless you need the full materialized result.
- Prefer generator expressions or iterators when one-pass consumption is enough.
- Avoid repeated `model_dump()` or `model_validate()` calls inside tight loops when batch validation or serialization can be used.
- Use `TypeAdapter(list[T])` for large collection validation instead of per-item model validation loops.

## 6. Cache carefully

- Cache expensive pure computations, but do not cache large objects blindly.
- Unbounded caches can turn a CPU optimization into a memory leak.
- Prefer bounded caches and explicit eviction strategy.
- Reuse heavyweight clients and connection pools from app lifespan instead of rebuilding them per request.

## 7. Worker count is a memory setting too

- More workers improve concurrency only up to a point.
- Every worker adds baseline memory overhead.
- Tune worker count together with:
  - container memory limit
  - preload strategy
  - allocator choice
  - request latency profile

## 8. Measure before and after

- Check RSS, not just Python object size.
- Benchmark with realistic concurrency.
- Separate:
  - idle memory
  - steady-state memory
  - peak memory during heavy responses
- Treat memory claims like "30% lower" as workload-dependent, not universal.


# Unit Testing

The Basics of Unit Testing (1:01): Validating the behavior of small, isolated pieces of code (functions/methods) to catch bugs, ensure safe refactoring, and document behavior.
Monkey Patching (3:45): Dynamically replacing functions at runtime (e.g., swapping real HTTP requests for fake ones) to make tests deterministic.
Mocking (8:51): Using unittest.mock (specifically MagicMock) to create flexible fake objects, which allows for advanced assertions like checking if a method was called.
Fixtures (12:20): Utilizing pytest.fixture to handle setup and teardown of test states, promoting code reuse.
Refactoring for Testability (14:01): Improving code design by introducing dependencies (like an HTTP client) to make testing easier without complex patching.
Advanced pytest Features (16:39):
Parameterization: Running the same test with different inputs using @pytest.mark.parametrize.
Exception Testing: Using pytest.raises to ensure code handles errors correctly.
Skipping/X-failing: Using @pytest.mark.skip or @pytest.mark.xfail to manage known issues or conditional testing.
Best Practices (19:59): Aim for a single assertion per test, keep test names descriptive, and maintain a clear file structure (tests/) separate from production code.

What the Host Says to Do (Best Practices)
Keep tests focused and small (1:01): Unit tests should validate a single, isolated piece of code, such as a function or method, to keep them fast and easy to run.
Use pytest instead of unittest (3:16): The host strongly recommends pytest because it allows for simpler function-based tests, powerful assertions, and a more pleasant user experience.
Use Monkey Patching for external dependencies (3:51): When your code calls an external service (like an API), use monkeypatch to replace the real function with a fake one (setattr(httpx, 'get', fake_get)) so your tests don't make actual network calls (3:54).
While monkey patching allows you to test existing, tightly coupled code, Arjan notes that the process is "ugly" and difficult to maintain. This serves as a precursor to the second part of the series, where he will demonstrate how refactoring (specifically using dependency injection) simplifies testing and yields a cleaner, more modular design.

Leverage MagicMock for complex objects (8:51): Use unittest.mock.MagicMock to create objects that mimic external APIs without needing to write custom fake classes. You can configure return values for methods like json or raise_for_status (10:03).
Utilize Fixtures for setup (12:20): Use @pytest.fixture to handle the repetitive setup and teardown of objects, making your test functions cleaner and more reusable (13:05).
Refactor for Testability (14:01): Improve your code design by using Dependency Injection (e.g., passing a client object to the service) rather than hardcoding external dependencies inside methods (15:05).
Parameterize tests (16:49): Use @pytest.mark.parametrize to run the same test logic with multiple different input data sets, avoiding code duplication (17:00).
Test for exceptions (17:45): Use pytest.raises to ensure your code correctly handles and raises expected errors (17:57).
Use Descriptive Naming (20:18): Name your tests clearly so that the intent is obvious when reading test reports (e.g., test_get_temperature_with_monkeypatch).
What Not to Do (Pitfalls)
Do not make real API calls in unit tests (0:18): Hardcoding HTTP requests makes tests slow, unreliable, and prone to failing due to network issues rather than code bugs.
Do not use unit tests to write sloppy code (2:37): Tests are not an excuse to skip proper software design; good design leads to code that is naturally easier to test.
Do not mix production and test code (20:33): Keep your tests in a separate directory (e.g., a tests/ folder) away from the source code (20:45).
Do not have multiple assertions per test (20:02): The host recommends focusing each test on a single, specific outcome, usually resulting in a single assert statement per test.
