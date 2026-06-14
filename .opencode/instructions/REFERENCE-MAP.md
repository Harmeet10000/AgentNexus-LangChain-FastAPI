# Reference Map

Use these files as the first place to look before inventing a new pattern.

- Lifespan reference: `src/app/lifecycle/lifespan.py`
- Global exception handler: `src/app/middleware/global_exception_handler.py`
- Logging reference: `src/app/examples/logger_usage_example.py`
- Celery task reference: `src/app/examples/CELERY.md`
- Cache reference: `src/app/utils/cache/redis_func.py`
- API response envelope: `src/app/shared/response_type.py`
- Examples belong in: `src/app/examples`

## Tooling and External References

- Use Context7 MCP server when docs are version-sensitive, unclear, or likely changed.
- Ask for agent skill when required and available in `.github/skills` and `.github/agents`.

## graphify

For any question about this repo's architecture, structure, components, or how to add/modify/find code, first use:
- `graphify query "<question>"` — when `graphify-out/graph.json` exists
- `graphify path "<A>" "<B>"` — for relationship questions
- `graphify explain "<concept>"` — for focused-concept questions

If `graphify-out/wiki/index.md` exists, use it for broad navigation.
Read `graphify-out/GRAPH_REPORT.md` only for broad architecture review.
Only read source files when modifying/debugging specific code or the graph lacks detail.

Type `/graphify` in Copilot Chat to build or update the graph.
