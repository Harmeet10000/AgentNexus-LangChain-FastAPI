
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


