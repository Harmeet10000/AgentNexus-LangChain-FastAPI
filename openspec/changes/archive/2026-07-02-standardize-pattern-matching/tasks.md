## 1. Documentation

- [x] 1.1 Add a reference to RESULT-PATTERN.md in `.opencode/instructions/ARCHITECTURE-RULES.md` under the service/repository rules section, pointing contributors to the pattern matching decision matrix

## 2. Redundant isinstance Removal — Batch 1: documents/service.py

- [x] 2.1 Remove `isinstance(cached, bytes)` check in `_cached_embedding()` (line 813) — replace with direct `.decode()` since Redis client is configured with decoder
- [x] 2.2 Remove `isinstance(raw_groups, list)` check in `_flatten_warnings()` (line 906) — the function parameter is typed as `object` but callers always pass `list`; tighten the parameter type to `list` instead
- [x] 2.3 Remove `isinstance(group, list)` and `isinstance(group, dict)` checks in `_flatten_warnings()` (lines 910, 916) — after tightening the parameter type, iterate directly
- [x] 2.4 Remove `isinstance(warning, dict)` check in `_flatten_warnings()` (line 914) — after parameter type is tightened, this is redundant
- [x] 2.5 Run `uv run ruff check src/app/features/documents/service.py` and `uv run ty check src/app/features/documents/service.py` to verify no regressions

## 3. Redundant isinstance Removal — Batch 2: agent_saul/service.py

- [x] 3.1 Audit isinstance checks in `_process_ws_message()` (lines 200-253) — these are on LangGraph state dicts and are LEGITIMATE; document why each is kept with a `# dynamic data` comment
- [x] 3.2 Audit isinstance checks on `WSPingMessage` / `WSResumeMessage` (lines 319, 323) — LEGITIMATE WebSocket discriminated union; keep as-is
- [x] 3.3 Run `uv run ruff check src/app/features/agent_saul/service.py` and `uv run ty check src/app/features/agent_saul/service.py`

## 4. Redundant isinstance Removal — Batch 3: shared utilities

- [x] 4.1 Remove `isinstance(data, BaseModel)` check in `utils/http_response.py` (line 13) — Pydantic model output is always a BaseModel
- [x] 4.2 Remove `isinstance(data, list)` check in `utils/http_response.py` (line 15) — the parameter is typed
- [x] 4.3 Remove `isinstance(key, bytes)` checks in `utils/cache/redis_func.py` (lines 70, 88, 111, 565) — audit which are on typed inputs vs Redis responses; keep only Redis-origin ones
- [x] 4.4 Remove `isinstance(value, list)` checks in `utils/cache/redis_func.py` (lines 927, 961, 967, 980, 1002, 1014, 1048, 1311, 1359) — audit which are on typed inputs; remove redundant ones
- [x] 4.5 Remove `isinstance(value, dict)` check in `utils/logger.py` (line 91) — if the parameter is typed, remove
- [x] 4.6 Run `uv run ruff check src/app/utils/` and `uv run ty check src/app/utils/`

## 5. Redundant isinstance Removal — Batch 4: auth and search services

- [x] 5.1 Remove `isinstance(exc.detail, dict)` and `isinstance(exc.detail, str)` checks in `middleware/global_exception_handler.py` (lines 36, 39, 86) — LEGITIMATE (external Starlette exception detail is untyped); keep but add comment
- [x] 5.2 Remove `isinstance(task_result.result, dict)` check in `search/service.py` (line 130) — LEGITIMATE (Celery result is dynamically typed); keep
- [x] 5.3 Run `uv run ruff check src/app/features/search/service.py src/app/middleware/global_exception_handler.py`

## 6. Replace isinstance+model_validate Hybrid

- [x] 6.1 In `features/ingestion/service.py` line 76, replace `failure if isinstance(failure, AppError) else AppError.model_validate(failure)` with match/case: `match failure: case AppError() as error: pass case dict() as raw: error = AppError.model_validate(raw) case _: error = AppError(code="UNKNOWN", message=str(failure))`
- [x] 6.2 Run `uv run ruff check src/app/features/ingestion/service.py` and `uv run ty check src/app/features/ingestion/service.py`

## 7. Convert Service-Layer isinstance to match/case (Optional Consistency)

- [x] 7.1 In `features/auth/service.py`, audit if any isinstance checks on Result types exist and convert to match/case (currently all already use match/case — verify)
- [x] 7.2 In `features/users/service.py`, verify the 2 match/case blocks are already consistent (they are — confirm no isinstance on Results)
- [x] 7.3 Run `uv run ruff check src/app/features/auth/service.py src/app/features/users/service.py`

## 8. Final Verification

- [x] 8.1 Run full lint: `uv run ruff check src/`
- [x] 8.2 Run full type check: `uv run ty check src/`
- [x] 8.3 Run tests: `uv run pytest tests/ -x -q`
- [x] 8.4 Verify RESULT-PATTERN.md is accurate by spot-checking 3 service files against the documented patterns
