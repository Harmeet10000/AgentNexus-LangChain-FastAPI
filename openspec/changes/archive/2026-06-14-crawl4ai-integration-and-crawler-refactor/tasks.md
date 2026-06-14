## 1. Fix Crawler Module Convention Violations (🔴)

- [x] 1.1 Fix import paths in `config.py` and `crawler.py`: change `from app.config.settings import get_settings` to `from app.config import get_settings`
- [x] 1.2 Fix `CrawlerConfig.model_validator` destructiveness: replace `model_validator(mode="before")` with per-field `Field(default_factory=...)` pattern that only overrides from settings when env vars are present, preserving model defaults
- [x] 1.3 Fix `CrawlResult.model_config`: change `{"arbitrary_types_allowed": True}` to `ConfigDict(arbitrary_types_allowed=True)` and add import
- [x] 1.4 Fix silent cache errors in `_get_from_cache` and `_save_to_cache`: replace `except Exception: pass` with `logger.bind(operation=...).exception(...)` and add `from app.utils import logger`
- [x] 1.5 Fix proxy gate on caching: remove the `if not settings.CRAWL4AI_PROXY: return None` guard from `_get_from_cache`

## 2. Fix Crawler Module Convention Violations (🟡)

- [x] 2.1 Refactor `GeminiProcessor.__init__` to accept optional `ChatGoogleGenerativeAI | None` parameter, defaulting to `_build_chat_model()` when None; move import to top level
- [x] 2.2 Remove redundant second-level character re-check loop in `smart_chunk_markdown` (lines 60-65 in chunker.py)
- [x] 2.3 Make `get_crawler()` accept an optional `redis_client` parameter to wire caching properly
- [x] 2.4 Run `uv run ruff format src/ && uv run ruff check --fix src/ && uv run ty check src/` to verify all fixes

## 3. Add Lifespan-Managed AsyncWebCrawler

- [x] 3.1 Add `AsyncWebCrawler` startup in FastAPI lifespan: create with `BrowserConfig(headless=True)`, call `.start()`, store in `app.state.crawl4ai_crawler`
- [x] 3.2 Add `AsyncWebCrawler` shutdown in FastAPI lifespan: call `await app.state.crawl4ai_crawler.close()` with `None` guard
- [x] 3.3 Wrap startup in try/except so crawler failure sets `app.state.crawl4ai_crawler = None` without blocking app boot

## 4. Wire Crawler into open_deep_search Config

- [x] 4.1 In `build_open_deep_search_config(config.py)`: read `crawl4ai_crawler` from `request.app.state` and add to `RunnableConfig.configurable`
- [x] 4.2 Add `_get_crawl4ai_crawler_from_config()` helper in `open_deep_search/utils.py` (mirrors `_get_httpx_client_from_config`)

## 5. Add crawl_webpage Tool and crawl_executor Node

- [x] 5.1 Create `crawl_webpage` `@tool` function in `open_deep_search/utils.py`: accepts single URL, validates URL, reads `AsyncWebCrawler` from config, calls `crawler.arun(url=url, config=...)`, formats result as markdown + metadata
- [x] 5.2 Add `crawl_webpage` to `get_all_tools()` with `metadata.type == "crawl"` and `metadata.name == "crawl_webpage"`
- [x] 5.3 Add `crawl_executor` node function in `graph.py`: checks for `crawl_webpage` tool calls, executes via `asyncio.gather`, returns `ToolMessage` results; short-circuits if no crawl calls present
- [x] 5.4 Register `crawl_executor` node in `researcher_builder`, add dual-conditional edge: `researcher ──► crawl_executor ──► (loop to researcher or compress_research)` parallel to existing `researcher_tools` edge

## 6. Update Prompts

- [x] 6.1 Update `_RESEARCH_SYSTEM_PROMPT` in `prompts.py`: add guidance on when to use `crawl_webpage` vs `web_search`
- [x] 6.2 Update `_RESEARCH_SYSTEM_PROMPT` execution policy constraints to mention crawl tool alongside search limits

## 7. Verify

- [x] 7.1 Run `uv run ruff format src/` — no formatting errors
- [x] 7.2 Run `uv run ruff check src/` — no lint errors
- [x] 7.3 Run `uv run ty check src/` — no type errors
- [x] 7.4 Verify no bare `except Exception: pass` remains in crawler module
