## Context

The project has a well-designed LLM factory at `shared/langchain_layer/models.py:88` (`_build_chat_model()`) that centralizes model configuration (temperature, top_p, top_k, max_tokens, API key). But two service modules bypass it:

- `search/service.py:236`: `ChatGoogleGenerativeAI(model=..., api_key=..., temperature=0.1, retries=0)`
- `documents/service.py:326,437`: `ChatGoogleGenerativeAI(model=..., api_key=..., temperature=0.1, retries=0)`

These create a new HTTP client pool per request, waste connections, and hardcode model params that might drift from settings.

For `datetime.utcnow()`: Python 3.12 deprecated this function. It returns a naive datetime (no timezone info) which breaks comparisons with timezone-aware datetimes. The codebase already uses `datetime.now(datetime.timezone.utc)` in most places — 7 stragglers remain.

## Goals / Non-Goals

**Goals:**
- Inject LLM client via constructor (dependency injection)
- Use existing `_build_chat_model()` factory or accept `BaseChatModel` as parameter
- Replace all `datetime.utcnow()` with `datetime.now(datetime.timezone.utc)`
- Zero breaking changes to API contracts

**Non-Goals:**
- Rewrite the entire LLM factory
- Add async LLM factory (current factory is sync, used in sync contexts)
- Change datetime handling beyond the 7 straggler locations

## Decisions

### D1: LLM injection via constructor, not factory

**Decision:** `SearchService.__init__` takes `llm: BaseChatModel` as a required parameter. The dependency injection layer (`dependencies.py`) creates the LLM once and passes it in.

**Rationale:** Constructor injection is the simplest DI pattern. The LLM is a long-lived resource (connection pool) that should be created once per app lifetime, not per request. This matches how `SearchService` already receives `repo` and `redis`.

**Alternatives considered:**
- *Use `_build_chat_model()` in service*: still creates per-request — rejected
- *Singleton LLM in module scope*: hard to test, hides dependency — rejected
- *FastAPI `Depends()` for LLM*: overkill for a non-request-scoped resource — rejected

### D2: datetime — direct replacement, no migration

**Decision:** Replace `datetime.utcnow()` → `datetime.now(datetime.timezone.utc)` at each call site. No data migration needed (existing naive datetimes in DB are fine — they're all UTC already).

**Rationale:** The existing DB values are UTC naive datetimes. Replacing the call sites makes new values timezone-aware. Comparisons between naive and aware datetimes will raise `TypeError` — but the codebase already stores UTC timestamps, so the comparison partners are also UTC. We need to check each comparison site.

**Alternatives considered:**
- *Add `import datetime; datetime.timezone = UTC`*: monkey-patching — rejected
- *Use `pendulum.now("UTC")`*: already a dependency, but adds coupling — rejected
- *Migrate all DB datetimes to aware*: overkill — rejected

## Risks / Trade-offs

- **[LLM injection breaks existing tests]** If any tests create `SearchService` without an LLM mock, they'll fail. **Mitigation:** Update test fixtures to provide a mock LLM.
- **[datetime comparison TypeError]** Mixing naive and aware datetimes raises `TypeError`. **Mitigation:** Check each comparison site — the DB stores naive UTC, so comparisons should be against naive UTC or aware UTC consistently.
- **[documents/service.py is complex]** Multiple functions create LLMs. **Mitigation:** Accept `llm` parameter in each function, inject at the dependency layer.
