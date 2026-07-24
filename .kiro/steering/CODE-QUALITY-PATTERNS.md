# Code Quality Patterns

Recommended Python stdlib patterns for this codebase.

## `functools.cache` over `@lru_cache(maxsize=1)`

Use `@functools.cache` for zero-argument, build-once-reuse-forever functions. Clearer semantics than `@lru_cache(maxsize=None)` or `@lru_cache(maxsize=1)`.

**Good fit:**
- No-arg factory functions in settings (e.g., `settings.py`)
- Singleton client getters (e.g., langextract `client.py`, agents `registry.py`)

**Keep `lru_cache`** when caching by argument (keyed clients, parameterized constructors).

## `itertools.pairwise`

Use `itertools.pairwise(seq)` instead of `for i in range(len(seq) - 1)` for adjacent-element iteration.

**Good fit:**
- Chunk overlap logic in `chunker.py`
- Edge/window traversal in `graphs.py`

## `Protocol` for structural typing

Use `typing.Protocol` at integration boundaries to reduce coupling to vendor SDK classes.

**Good fit:**
- Search embedding interface in search `service.py`
- Agent/tool registry contracts
- Crawler adapter interfaces
- Vector store abstractions

Benefits: easier mocking, cleaner DI, no forced inheritance from SDK bases.

## `match` with guards

Use `match`/`case` with guards for branchy enum/mode/type dispatchers. More declarative than if/elif chains.

**Good fit:**
- Crawl mode dispatch in crawler `service.py`
- Response shaping in `httpResponse.py`
- Content-type classification in `docling_enhanced.py`

## `pathlib` standardization

Prefer `pathlib.Path` over `os.path` everywhere. Ruff `PTH` rule enforces this.

**Areas still using `os.path`:**
- `ingest.py` (document processing)
- `multimodal/code.py` (rag)

Convert string-path operations (`os.path.join`, `os.path.split`, `os.path.exists`) to `Path` equivalents.

## `contextlib.suppress`

Use `contextlib.suppress(SomeException)` for intentionally ignored errors (best-effort cleanup, optional deletes). More explicit than bare `except: pass`.

Currently limited applicability; introduce naturally as cleanup code grows.

## `contextlib.AsyncExitStack`

Prefer `AsyncExitStack` over `ExitStack` for async resource lifecycle management (lifespan, ingestion orchestration). Use when resource lifetimes are genuinely dynamic (conditionally acquired).
