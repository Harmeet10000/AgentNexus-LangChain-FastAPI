## Context

The project uses loguru for structured logging (`from loguru import logger`). Two files in `src/app/shared/rag/` contain legacy `print()` calls from early prototyping — one in a CLI chat script (`rag_agent_advanced.py`, 17 calls) and one in an ingestion summary script (`ingest.py`, 13 calls). These bypass the logging pipeline entirely.

## Goals / Non-Goals

**Goals:**
- Replace all 30 `print()` calls with structured loguru equivalents
- Preserve user-facing output readability in CLI scripts
- Zero behavioral change — output destination shifts from stdout to log sink

**Non-Goals:**
- Restructuring the CLI scripts themselves
- Adding new log sinks or formatters
- Changing log levels elsewhere in the codebase

## Decisions

1. **Use `logger.info()` for user-facing output, `logger.debug()` for progress/detail**
   - Rationale: CLI scripts print status for the operator; `info` is the right level. Detailed progress (chunk counts, iteration status) goes to `debug`.
   - Alternative considered: Use `print()` for user-facing and `logger` for internal — rejected because it creates two output paths to manage.

2. **Keep string formatting inline, don't switch to `logger.bind()` for these scripts**
   - Rationale: These are simple CLI scripts, not request-scoped services. Binding context like `request_id` is unnecessary here.

3. **For streaming output in `rag_agent_advanced.py` (the `end=""` flush pattern), use `logger.info()` without `end=` — loguru doesn't support `end=""`**
   - Rationale: Streaming token-by-token output doesn't map to structured logging. Use `logger.info(text, end="")` won't work — instead, buffer the full response and log once.

## Risks / Trade-offs

- **Risk**: CLI users expecting stdout output may not see log messages without configuring loguru's stderr sink. **Mitigation**: loguru defaults to stderr; CLI scripts run in terminal where stderr is visible.
- **Risk**: Streaming output (`end="", flush=True`) is lost. **Mitigation**: Log the complete response after streaming finishes.
