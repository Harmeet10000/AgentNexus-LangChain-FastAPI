# Crawl4AI Configuration Spec

**Date:** 2026-06-21
**Scope:** `src/app/shared/crawler/` — config, crawler, connections
**Status:** Gaps identified, specs proposed

---

## 1. Current State Summary

Production crawler has 3 independent instantiation paths with inconsistent config:

| Path | Config Used | What's Missing |
|---|---|---|
| `connections/crawl4ai.py` (lifespan singleton) | `BrowserConfig(headless=True)` only | timeout, user-agent, sandbox flags, proxy |
| `crawler.py` → `WebCrawler.crawl()` | Full `CrawlerConfig.to_browser_config()` | excluded_tags, content filters, wait_for |
| `open_deep_search/utils.py` (LangGraph tool) | Bare `CrawlerRunConfig` | Everything — no browser config at all |

**`CrawlerConfig` (config.py) has 11 fields.** Crawl4AI exposes ~60+ `BrowserConfig` + `CrawlerRunConfig` options. The config covers ~18% of what matters.

---

## 2. Missing Specs

### SPEC-01: Content Noise Filtering

**Problem:** Raw markdown includes nav bars, footers, sidebars, cookie banners, ads. No `excluded_tags`, no content filters, no `word_count_threshold`.

**Production code:** `to_crawler_run_config()` returns only `cache_mode` and `stream`. No `excluded_tags`, no `DefaultMarkdownGenerator` options, no `PruningContentFilter`.

**Example (02.2):** `excluded_tags=["nav", "footer", "header"]`
**Example (05.2):** `PruningContentFilter(threshold=0.5, min_word_threshold=50)`

**Proposed fields on `CrawlerConfig`:**

```python
excluded_tags: list[str] = ["nav", "footer", "header", "aside", "form"]
word_count_threshold: int = 30
pruning_threshold: float = 0.5
pruning_min_words: int = 50
ignore_links: bool = True
ignore_images: bool = True
```

**Impact:** 30-60% reduction in markdown token count. Cleaner LLM input.

---

### SPEC-02: Lifespan Crawler Config Parity

**Problem:** `create_crawl4ai_crawler()` passes only `headless=True`. Ignores timeout, user-agent, sandbox flags, proxy, extra_args.

**File:** `src/app/connections/crawl4ai.py:26-28`

**Current:**
```python
crawler = AsyncWebCrawler(
    config=BrowserConfig(headless=settings.CRAWL4AI_HEADLESS),
)
```

**Should be:**
```python
crawler = AsyncWebCrawler(
    config=BrowserConfig(**get_crawler_config().to_browser_config()),
)
```

**Impact:** Lifespan crawler (used by LangGraph `crawl_webpage` tool) currently runs with default timeout, no user-agent spoofing, no sandbox hardening.

---

### SPEC-03: BrowserConfig Viewport & Identity

**Problem:** No viewport size control, no user-agent randomization, no stealth mode.

**Example (02.1):** `viewport_width=1440, viewport_height=900, user_agent_mode="random"`
**Example (18):** `enable_stealth=True`
**Production:** User-agent hardcoded via `--user-agent=` extra_args, no viewport, no stealth.

**Proposed fields on `CrawlerConfig`:**

```python
viewport_width: int = 1280
viewport_height: int = 720
enable_stealth: bool = False
user_agent_mode: str = "fixed"  # "fixed" | "random"
```

**Note:** `user_agent_mode="random"` rotates UA per crawl. Useful for avoiding fingerprinting on anti-bot sites. Default `"fixed"` preserves current behavior.

---

### SPEC-04: Page Interaction & Wait Strategy

**Problem:** No `wait_for`, no `wait_until`, no `page_timeout` on `CrawlerRunConfig`. JS-heavy pages may return incomplete content.

**Example (02.2):** `wait_for="css:main"`, `wait_until="networkidle"`, `page_timeout=30_000`
**Example (12.1-12.2):** `js_code` for scroll/click interactions
**Production:** `to_crawler_run_config()` returns no wait or timeout config.

**Proposed fields on `CrawlerConfig`:**

```python
page_timeout: int = 30_000  # ms
wait_until: str | None = None  # "networkidle" | "load" | "domcontentloaded"
wait_for: str | None = None  # CSS selector or JS expression
```

**Proposed method `to_crawler_run_config()` expansion:**

```python
def to_crawler_run_config(self) -> dict[str, Any]:
    config = {
        "cache_mode": self.cache_mode,
        "stream": False,
        "word_count_threshold": self.word_count_threshold,
        "excluded_tags": self.excluded_tags,
        "exclude_external_links": True,
        "exclude_social_media_links": True,
    }
    if self.page_timeout:
        config["page_timeout"] = self.page_timeout
    if self.wait_until:
        config["wait_until"] = self.wait_until
    if self.wait_for:
        config["wait_for"] = self.wait_for
    return config
```

---

### SPEC-05: MarkdownGenerator with Content Filters

**Problem:** `DefaultMarkdownGenerator()` is used without options or content filters. Output includes link clutter, boilerplate, and irrelevant sections.

**Example (05.2):** `PruningContentFilter(threshold=0.5, min_word_threshold=50)`
**Example (24):** `BM25ContentFilter(user_query=..., bm25_threshold=1.0)` for query-focused extraction
**Production:** `markdown_generator=DefaultMarkdownGenerator()` — no filter, no options.

**Proposed helper on `CrawlerConfig`:**

```python
def get_markdown_generator(
    self,
    content_filter: PruningContentFilter | BM25ContentFilter | None = None,
) -> DefaultMarkdownGenerator:
    options = {
        "ignore_links": self.ignore_links,
        "ignore_images": self.ignore_images,
    }
    return DefaultMarkdownGenerator(content_filter=content_filter, options=options)
```

**Usage in `WebCrawler.crawl()`:**
```python
from crawl4ai.content_filter_strategy import PruningContentFilter

content_filter = PruningContentFilter(
    threshold=self.config.pruning_threshold,
    min_word_threshold=self.config.pruning_min_words,
)
run_config = CrawlerRunConfig(
    markdown_generator=self.config.get_markdown_generator(content_filter),
    **self.config.to_crawler_run_config(),
)
```

**Impact:** `fit_markdown` field becomes available — pruned, focused content. 30-60% smaller markdown for LLM consumption.

---

### SPEC-06: Recursive Crawl with Streaming & URL Matching

**Problem:** `crawl_recursive()` does manual BFS loop. Crawl4AI has `BFSDeepCrawlStrategy` with streaming, URL pattern filters, and rate limiting built-in.

**Example (19.1):** `BFSDeepCrawlStrategy(max_depth=2, include_external=False)`
**Example (20.2):** `MemoryAdaptiveDispatcher` with `RateLimiter`
**Production (24):** `url_matcher=["*api/*"], match_mode=MatchMode.OR` for per-URL-type configs

**Current `crawl_recursive()`** (`crawler.py:205-293`): 90-line manual BFS with visited set, URL normalization, depth iteration. No streaming, no rate limiting, no URL matching.

**Proposed refactor:** Replace with crawl4ai's native deep crawl:

```python
async def crawl_recursive(
    self,
    urls: list[str],
    max_depth: int = 1,
    max_pages: int = 10,
) -> list[CrawlResult]:
    from crawl4ai.deep_crawl import BFSDeepCrawlStrategy

    browser_config = BrowserConfig(**self.config.to_browser_config())
    run_config = CrawlerRunConfig(
        deep_crawl_strategy=BFSDeepCrawlStrategy(
            max_depth=max_depth,
            max_pages=max_pages,
            include_external=False,
        ),
        **self.config.to_crawler_run_config(),
    )
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=self.config.memory_threshold,
        max_session_permit=self.config.max_concurrent,
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        results = await crawler.arun_many(urls, config=run_config, dispatcher=dispatcher)
    return [self._to_crawl_result(r) for r in results]
```

**Impact:** Eliminates 90 lines of manual BFS. Gains streaming, rate limiting, and URL pattern matching for free.

---

### SPEC-07: Rate Limiter on Dispatcher

**Problem:** `MemoryAdaptiveDispatcher` has no `RateLimiter`. Recursive crawl hits servers with no delay between requests.

**Example (24):** `RateLimiter(base_delay=(0.5, 1.0), max_retries=2)`
**Production (236-240):** `MemoryAdaptiveDispatcher` without rate limiter.

**Proposed field on `CrawlerConfig`:**

```python
rate_limit_delay: tuple[float, float] = (0.5, 1.0)  # min/max delay in seconds
rate_limit_retries: int = 2
```

**Proposed dispatcher construction:**

```python
dispatcher = MemoryAdaptiveDispatcher(
    memory_threshold_percent=self.config.memory_threshold,
    max_session_permit=self.config.max_concurrent,
    rate_limiter=RateLimiter(
        base_delay=self.config.rate_limit_delay,
        max_retries=self.config.rate_limit_retries,
    ),
)
```

---

### SPEC-08: URL-Specific Configs via `url_matcher`

**Problem:** Same config applied to all URLs. API docs, blog posts, and landing pages have different content structures.

**Example (24):** Three separate `CrawlerRunConfig` objects with `url_matcher` patterns routed via `match_mode=MatchMode.OR`.

**Proposed method on `CrawlerConfig`:**

```python
def to_crawler_run_configs(self) -> list[CrawlerRunConfig]:
    """Return URL-specific configs when url_patterns are set, else single default."""
    if not self.url_patterns:
        return [CrawlerRunConfig(**self.to_crawler_run_config())]

    configs = []
    for pattern in self.url_patterns:
        configs.append(CrawlerRunConfig(
            url_matcher=[pattern],
            match_mode=MatchMode.OR,
            **self.to_crawler_run_config(),
        ))
    return configs
```

**Proposed field:**

```python
url_patterns: list[str] = []  # glob patterns for URL-specific routing
```

---

## 3. Environment Variables to Add

| Variable | Default | Purpose |
|---|---|---|
| `CRAWL4AI_EXCLUDED_TAGS` | `"nav,footer,header,aside,form"` | Comma-separated HTML tags to strip |
| `CRAWL4AI_WORD_COUNT_THRESHOLD` | `30` | Min words per content block |
| `CRAWL4AI_PRUNING_THRESHOLD` | `0.5` | PruningContentFilter threshold |
| `CRAWL4AI_PAGE_TIMEOUT` | `30000` | Page load timeout (ms) |
| `CRAWL4AI_WAIT_UNTIL` | `None` | Wait strategy: networkidle/load/domcontentloaded |
| `CRAWL4AI_WAIT_FOR` | `None` | CSS selector to wait for |
| `CRAWL4AI_VIEWPORT_WIDTH` | `1280` | Browser viewport width |
| `CRAWL4AI_VIEWPORT_HEIGHT` | `720` | Browser viewport height |
| `CRAWL4AI_STEALTH` | `False` | Enable stealth mode |
| `CRAWL4AI_RATE_LIMIT_DELAY_MIN` | `0.5` | Min delay between requests (s) |
| `CRAWL4AI_RATE_LIMIT_DELAY_MAX` | `1.0` | Max delay between requests (s) |

---

## 4. Bug: Lifespan Singleton vs Per-Call Browser

The lifespan creates a long-lived `AsyncWebCrawler` (stored on `app.state`). `WebCrawler.crawl()` creates a fresh `AsyncWebCrawler` per call via context manager. Two independent browser instances may be active simultaneously.

**Recommendation:** Either:
- (a) Make `WebCrawler` use the lifespan-managed crawler instead of creating its own, OR
- (b) Remove the lifespan crawler entirely and let LangGraph tool use `WebCrawler` directly

Option (a) is cleaner. Pass the lifespan crawler into `WebCrawler.__init__()` and skip context manager creation in `crawl()`.

---

## 5. Priority Order

| # | Spec | Effort | Impact |
|---|---|---|---|
| 1 | SPEC-02: Lifespan parity | 5 min | Fixes silent misconfiguration |
| 2 | SPEC-01: Content noise | 15 min | 30-60% less markdown noise |
| 3 | SPEC-05: MarkdownGenerator filters | 15 min | `fit_markdown` for LLM input |
| 4 | SPEC-04: Wait strategy | 10 min | JS-heavy pages render correctly |
| 5 | SPEC-07: Rate limiter | 5 min | Polite crawling, no hammering |
| 6 | SPEC-03: Viewport & stealth | 10 min | Anti-bot bypass when needed |
| 7 | SPEC-06: Native deep crawl | 30 min | Remove 90-line manual BFS |
| 8 | SPEC-08: URL-specific configs | 20 min | Per-URL-type extraction |

**Total estimated effort:** ~2 hours for all specs. Specs 1-5 are the high-value core.
