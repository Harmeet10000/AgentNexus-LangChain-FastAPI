# Project Snapshot

- Project: `langchain-fastapi-production`
- Python: `3.12`
- Package manager: `uv`
- Formatter/linter: `ruff`
- Type checker: `ty`
- Framework stack: `FastAPI`, `Pydantic v2`, `LangChain`, `LangGraph`, `SQLAlchemy`, `Beanie`, `Redis`, `Celery`
- Architecture: modular monolith, feature-driven, async-first

## Logging, Errors, and Responses

- Use structured logging consistently (loguru).
- Keep logs contextual and machine-parseable.
- Use typed exceptions from `src/app/utils/exceptions.py`.
