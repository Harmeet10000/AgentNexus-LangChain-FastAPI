## Context

The open_deep_search graph uses a two-level hierarchy: an outer supervisor graph that delegates research topics to a researcher subgraph. The researcher subgraph currently has three tools: `tavily_search`, `think_tool`, and `ResearchComplete`. Tavily provides pre-summarized search results including `raw_content`, but cannot render JavaScript-heavy SPAs and returns only the content Tavily's crawler chose to fetch.

Crawl4AI is already a declared dependency in `pyproject.toml` with settings in `app.config.settings`. The `src/app/shared/crawler/` module wraps Crawl4AI but has several convention violations that need fixing before it can serve as a reliable graph tool.

The `_build_chat_model` factory in `models.py` is the project's standard for LLM instantiation — `GeminiProcessor` creates its own `ChatGoogleGenerativeAI` inline, bypassing this.

## Goals / Non-Goals

**Goals:**
- Fix all 🔴 convention violations in the crawler module
- Make `GeminiProcessor` use `_build_chat_model` via dependency injection
- Wire an `AsyncWebCrawler` instance into FastAPI lifespan (startup/shutdown)
- Pass the crawler instance through `RunnableConfig` via `build_open_deep_search_config`
- Add a `crawl_webpage` tool bound to the researcher LLM
- Add a dedicated `crawl_executor` node that routes only Crawl4AI tool calls
- Update the researcher system prompt to teach when to use `crawl_webpage`
- Keep the existing `researcher_tools` node unchanged (it still routes search/think/complete)

**Non-Goals:**
- No change to the supervisor subgraph or outer graph structure
- No change to Tavily search behavior or result format
- No introduction of `returns.Result` inside graph nodes (per project rules, node entrypoints raise exceptions)
- No change to the `crawl_recursive` method (not used by the graph tool)

## Decisions

### Decision 1: Tool + dedicated executor node over auto-enrichment

**Choice:** Add `crawl_webpage` as a LangChain `@tool` bound to the researcher LLM, routed through a dedicated `crawl_executor` node.

**Rationale:**
- The researcher LLM already decides which tools to call. Adding `crawl_webpage` follows the same pattern as `tavily_search`.
- A dedicated `crawl_executor` node (separate from `researcher_tools`) allows Crawl4AI-specific error handling, timeout management, and content processing without affecting the generic tool executor.
- The LLM can decide when a URL from search results warrants deeper crawling — better resource utilization than auto-enrichment.
- Separation of concerns: `researcher_tools` handles the generic search/think/complete loop; `crawl_executor` handles browser-based page loading.

**Alternatives considered:** Post-search auto-enrichment (wastes resources on URLs already well-served by Tavily's `raw_content`), pre-search crawl (limited applicability).

### Decision 2: Lifespan-managed AsyncWebCrawler

**Choice:** Initialize `AsyncWebCrawler` in FastAPI lifespan, store in `app.state.crawl4ai_crawler`, shut down on app shutdown.

**Rationale:**
- Crawl4AI's `AsyncWebCrawler` manages a browser process — creating it per-request is expensive (~500ms+ startup).
- Matches the Tavily pattern where `httpx_client` is lifespan-managed and injected via `RunnableConfig`.
- Enables connection reuse and browser process pooling.

**Implementation pattern:**
```python
# lifespan.py startup
from crawl4ai import AsyncWebCrawler
from crawl4ai import BrowserConfig
crawler = AsyncWebCrawler(config=BrowserConfig(headless=True))
await crawler.start()
app.state.crawl4ai_crawler = crawler

# lifespan.py shutdown
await app.state.crawl4ai_crawler.close()
```

### Decision 3: Refactor GeminiProcessor to accept model dependency

**Choice:** Accept `ChatGoogleGenerativeAI | None` in `GeminiProcessor.__init__`, defaulting to `_build_chat_model()` when None.

**Rationale:**
- Removes I/O from constructor (violates project rules).
- Allows callers to inject the same model instance used elsewhere in the graph, reducing redundant client creation.
- Follows the project convention: "Initializers should not perform I/O, DB access, or heavy computation. Delegate to factory functions or class methods."

### Decision 4: Fix CrawlerConfig.model_validator to merge instead of replace

**Choice:** Replace `model_validator(mode="before")` with per-field `Field(default_factory=...)` or `model_validator(mode="after")` that only overrides from non-None settings values.

**Rationale:**
- Current implementation unconditionally replaces all fields with settings values, ignoring model defaults.
- `Field(default_factory=...)` is the established pattern from `open_deep_search/config.py:35-37`.
- After-validator prevents `None` propagation from missing settings.

### Decision 5: Dedicated graph edge for crawl_executor

**Choice:** Insert `crawl_executor` node between the researcher LLM output and the loop-back decision, parallel to `researcher_tools`.

```
researcher ──► crawl_executor ──► (loop to researcher or compress_research)
researcher ──► researcher_tools ──► (loop to researcher or compress_research)
```

**Rationale:**
- The researcher LLM may call both `tavily_search` and `crawl_webpage` in the same turn. Each tool type routes to its own executor.
- Crawl4AI calls have different error modes (browser crash, timeout, JS failure) than Tavily calls (network error, rate limit).
- If neither tool type is called, both nodes short-circuit to `compress_research`.

## Risks / Trade-offs

- **Risk**: Crawl4AI browser process consumes significant memory (~200-500MB). **Mitigation**: Monitor RSS in production; add memory threshold config matching `MemoryAdaptiveDispatcher` limits.
- **Risk**: Crawl4AI startup time adds to app cold start. **Mitigation**: Lazy-init on first crawl request instead of lifespan start if cold start is problematic.
- **Risk**: Adding a tool increases LLM decision complexity — the researcher may over-use or under-use crawling. **Mitigation**: Prompt engineering in `_RESEARCH_SYSTEM_PROMPT` with clear when-to-crawl guidance.
- **Trade-off**: Tool-based approach means the researcher LLM pays token cost for deciding whether to crawl. Acceptable because LLM judgment beats auto-crawl waste.
- **Trade-off**: The `_get_from_cache` method checks `settings.CRAWL4AI_PROXY` as a proxy for cache-enabled. This is fixed by removing that guard — caching should be unconditional when Redis is available.

## Open Questions

- Should the crawl tool accept a single URL or a list of URLs (like `tavily_search` accepts `queries: list[str]`)? Single URL keeps the tool focused; a batch mode could be added later.
- Should crawl results be cached in Redis or use Crawl4AI's built-in `cache_mode`? Initial implementation uses Crawl4AI's `cache_mode: "bypass"` (current default) since the graph tool is ephemeral.
