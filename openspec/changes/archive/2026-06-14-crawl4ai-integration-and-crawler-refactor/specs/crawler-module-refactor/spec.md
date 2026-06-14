## ADDED Requirements

### Requirement: Import paths use app.config not app.config.settings
All files in `src/app/shared/crawler/` SHALL import `get_settings` from `app.config`, not from `app.config.settings`.

#### Scenario: config.py uses correct import
- **WHEN** `src/app/shared/crawler/config.py` is read
- **THEN** it SHALL contain `from app.config import get_settings` instead of `from app.config.settings import get_settings`

#### Scenario: crawler.py uses correct import
- **WHEN** `src/app/shared/crawler/crawler.py` is read
- **THEN** it SHALL contain `from app.config import get_settings` instead of `from app.config.settings import get_settings`

### Requirement: CrawlerConfig does not destroy model defaults
The `CrawlerConfig.model_validator` SHALL NOT unconditionally replace all field values with settings values. It SHALL use `Field(default_factory=...)` or a `model_validator(mode="after")` that only overrides from non-None settings.

#### Scenario: Default values survive when settings are absent
- **WHEN** `CrawlerConfig()` is created with no arguments
- **THEN** `headless` SHALL default to `True`, `timeout` to `30000`, `cache_mode` to `"bypass"`, and all other fields to their model defaults

#### Scenario: Settings override defaults when present
- **WHEN** environment variable `CRAWL4AI_HEADLESS=false` is set
- **THEN** `CrawlerConfig().headless` SHALL be `False`

### Requirement: CrawlResult.model_config uses ConfigDict
`CrawlResult.model_config` SHALL use `ConfigDict(arbitrary_types_allowed=True)` instead of a raw dictionary.

#### Scenario: model_config is a ConfigDict
- **WHEN** `CrawlResult.model_config` is inspected
- **THEN** it SHALL be an instance of `ConfigDict`

### Requirement: Cache errors are logged, not silently swallowed
The `_get_from_cache` and `_save_to_cache` methods in `WebCrawler` SHALL log cache errors using `logger.bind(...).exception(...)` instead of bare `except Exception: pass`.

#### Scenario: Cache read failure is logged
- **WHEN** `_get_from_cache` encounters a Redis error
- **THEN** the error SHALL be logged with `logger.bind(operation="cache_read").exception(...)`

#### Scenario: Cache write failure is logged
- **WHEN** `_save_to_cache` encounters a Redis error
- **THEN** the error SHALL be logged with `logger.bind(operation="cache_write").exception(...)`

### Requirement: Caching is not gated by proxy setting
The `_get_from_cache` method SHALL NOT check `settings.CRAWL4AI_PROXY` as a condition for reading from cache.

#### Scenario: Cache works without proxy
- **WHEN** `CRAWL4AI_PROXY` is not set but Redis is available
- **THEN** `_get_from_cache` SHALL attempt to read from Redis without checking the proxy setting

### Requirement: GeminiProcessor accepts model via dependency injection
`GeminiProcessor.__init__` SHALL accept an optional `ChatGoogleGenerativeAI | None` parameter, defaulting to `_build_chat_model()` when `None`.

#### Scenario: Model injected explicitly
- **WHEN** `GeminiProcessor(model=my_model)` is called
- **THEN** the processor SHALL use `my_model` for all LLM calls

#### Scenario: No model falls back to _build_chat_model
- **WHEN** `GeminiProcessor()` is called with no arguments
- **THEN** the processor SHALL create a model using `_build_chat_model()` from `app.shared.langchain_layer.models`

### Requirement: smart_chunk_markdown removes redundant re-check loop
The second-level loop in `smart_chunk_markdown` (lines 60-65) that re-checks already-guaranteed chunk sizes SHALL be removed.

#### Scenario: No redundant character check
- **WHEN** `smart_chunk_markdown` processes content
- **THEN** each chunk SHALL be split at most once by character count after header-based splitting
