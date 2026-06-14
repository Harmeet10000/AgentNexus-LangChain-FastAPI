## Why

The `src/app/shared/crawler/` module has several convention violations (import paths, Pydantic config patterns, bare exception swallowing, I/O in constructors) that need alignment with the project's established patterns. Separately, the open_deep_search graph currently relies solely on Tavily's pre-summarized search results — adding crawl4ai as a researcher tool enables deeper content extraction from specific URLs, especially JavaScript-heavy pages Tavily cannot render fully.

## What Changes

- **Fix crawler module convention violations** — import paths, `CrawlerConfig.model_validator` destructiveness, `CrawlResult.model_config`, silent cache errors, redundant chunker logic, and `GeminiProcessor` model construction
- **Add lifespan-managed `AsyncWebCrawler`** — initialize in FastAPI lifespan, store in `app.state`, pass through `RunnableConfig`
- **Add `crawl_webpage` LangChain tool** bound to the researcher LLM in the open_deep_search researcher subgraph
- **Add dedicated `crawl_executor` graph node** — routes Crawl4AI tool calls separately from generic `researcher_tools`
- **Update prompts** to teach the researcher LLM when to use `crawl_webpage` vs `web_search`
- **Refactor `GeminiProcessor`** to accept model via dependency injection using `_build_chat_model` from `models.py`

## Capabilities

### New Capabilities
- `crawl4ai-graph-integration`: Crawl4AI tool binding, dedicated crawl_executor node, and lifespan wiring in the open_deep_search graph
- `crawler-module-refactor`: Convention fixes across all files in `src/app/shared/crawler/`

### Modified Capabilities
- `open-deep-search`: Researcher subgraph gains a new tool (`crawl_webpage`) and a new node (`crawl_executor`); tool assembly function `get_all_tools()` and researcher prompt are updated

## Impact

- `src/app/shared/crawler/` — fixes to `config.py`, `crawler.py`, `processor.py`, `chunker.py`
- `src/app/shared/langgraph_layer/open_deep_search/` — new tool + node in `graph.py`, updated `utils.py` tool assembly, updated prompts
- `src/app/lifecycle/lifespan.py` — register `AsyncWebCrawler` startup/shutdown
- `src/app/shared/langgraph_layer/open_deep_search/config.py` — add `build_open_deep_search_config` wiring for crawl4ai client
- `pyproject.toml` — no new dependencies (crawl4ai already declared)
