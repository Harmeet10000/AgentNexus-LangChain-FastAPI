## Why

30 `print()` calls exist in `src/app/shared/rag/rag_agent_advanced.py` and `src/app/shared/rag/document_processing/ingest.py`. These bypass the project's structured logging stack (loguru), making production debugging impossible — no log levels, no structured context, no correlation IDs, and output goes to stdout instead of the configured log sink.

## What Changes

- Replace all `print()` calls with appropriate `logger.info()`, `logger.debug()`, or `logger.error()` calls in the two affected files
- Add `from loguru import logger` imports where missing
- Preserve existing user-facing output in CLI scripts (the `__main__` blocks) by using `logger.info()` with clear formatting
- No API or behavioral changes — purely internal observability improvement

## Capabilities

### New Capabilities
- `print-to-logger`: Replaces all print() calls in app code with structured loguru logging

### Modified Capabilities

_(none — no existing specs change requirements)_

## Impact

- **Files**: `src/app/shared/rag/rag_agent_advanced.py`, `src/app/shared/rag/document_processing/ingest.py`
- **Dependencies**: loguru already in use — no new dependencies
- **APIs**: None
- **Risk**: Low — output destination changes from stdout to log sink; CLI scripts may need log level configuration
