# Pre-existing Test Baseline

Established 2026-08-17. Purpose: let the verifier leg distinguish **introduced** from **pre-existing** failures.
No source code was modified to produce this report.

## Headline

Command run (twice, identical results — suite is deterministic, zero flake):

```bash
uv run pytest -q --no-header -p no:cacheprovider
```

Exact summary line:

```
22 failed, 55 passed, 11 warnings, 13 errors in 18.63s
```

Re-run: `22 failed, 55 passed, 11 warnings, 13 errors in 16.18s`. Exit code **1**.

Nothing was narrowed, ignored, or timed out. The whole suite runs in under 20 seconds.
`pytest-timeout` is NOT installed (not needed — no test hangs).

> The earlier record of `22 failed, 41 passed, 13 errors` is stale only in the passed count.
> 55 vs 41 = the 14 `tests/unit/billing/` tests added by commit e2912e8 (Razorpay billing). Failures and
> errors are unchanged at 22/13.

### THE HEADLINE ANSWER: not one failure is environmental

The question this baseline was commissioned to answer — "missing local services or real breakage?" — resolves
cleanly to **real breakage**. Evidence, three independent ways:

1. `grep -icE 'connection refused|ConnectionError|ConnectTimeout|getaddrinfo|ServerSelectionTimeout|Neo4jError|NoCredentials|EndpointConnection'` over the full output returns **0**.
2. `grep -icE 'ModuleNotFoundError|ImportError|cannot import name'` returns **0**. There are **no collection
   errors of the import kind** — the bucket the brief expected to be most diagnostic is empty.
3. No test file references any service endpoint. `grep -rlE 'localhost|127\.0\.0\.1|:5432|:6379|:27017|:7687'`
   across every file in `tests/` returns **nothing**.

The suite is **hermetic by construction**: Redis is `fakeredis.aioredis.FakeRedis`, and everything else is
`MagicMock` — `tests/conftest.py:15-47` stubs 22 modules into `sys.modules` before any app module loads.
Starting Postgres/Redis/Mongo/Neo4j would change **nothing**. A verifier must NOT wave any of these away as
environmental.

## Taxonomy

| Bucket | Count | Verdict | Representative IDs + first error line |
|---|---|---|---|
| **Missing test fixture** (`client` never defined) | **13** (all "errors") | must-fix (test harness) | `tests/integration/test_health.py::TestHealthEndpoint::test_healthy_returns_200` — `E fixture 'client' not found`<br>`tests/integration/test_api_deprecation.py::TestDeprecationHeaders::test_v1_route_has_deprecation_header` — `E fixture 'client' not found` |
| **Stale test — mock returns raw value, prod code expects `Result`** | **10** | must-fix (update tests, prod is correct) | `tests/integration/test_search.py::TestSearchQuery::test_hybrid_search_finds_by_keyword` — `E AttributeError: 'list' object has no attribute 'unwrap'`<br>`tests/integration/test_auth.py::TestAuthRegister::test_register_creates_user` — `E AttributeError: 'bool' object has no attribute 'unwrap'`<br>`tests/integration/test_auth.py::TestAuthPasswordReset::test_reset_password_invalid_token_raises` — `E AttributeError: 'NoneType' object has no attribute 'unwrap'` |
| **GENUINE PRODUCTION BUG** — `logger.warning` on a module object | **6** | **must-fix (source, 1 line)** | `tests/unit/documents/test_normalize_embedding.py::test_truncates_oversized` — `E AttributeError: module 'app.utils.logger' has no attribute 'warning'` (all 6 identical) |
| **Test fixture data invalid** (fake argon2 hash) | **2** | must-fix (test harness) | `tests/integration/test_auth.py::TestAuthLogin::test_login_disabled_account_raises_unauthorized` — `E argon2.exceptions.InvalidHashError`<br>`...::test_login_unverified_email_raises_unauthorized` — same |
| **Test written against a symbol that no longer exists** | **1** | delete-or-rewrite | `tests/integration/test_auth.py::TestAuthRefresh::test_refresh_valid_token` — `E AttributeError: Mock object has no attribute 'find_by_id_result'` |
| **Under-specified MagicMock** (mock leaks into await / comparison) | **2** | must-fix (test harness) | `tests/integration/test_search.py::TestSearchIngestion::test_content_hash_dedup` — `E TypeError: object MagicMock can't be used in 'await' expression`<br>`tests/integration/test_auth.py::TestAuthPasswordReset::test_reset_password_updates_password` — `E TypeError: '<' not supported between instances of 'MagicMock' and 'datetime.datetime'` |
| **Genuine assertion failure** | **1** | must-fix | `tests/integration/test_auth.py::TestAuthPasswordReset::test_forgot_password_generates_token` — `E AssertionError: assert None is not None` / `+ where None = <MagicMock name='mock.find_by_email()'>.reset_token_hash` |
| **Environmental / service unavailable** | **0** | — | none exist |
| **Missing config / env var** | **0** | — | none exist (`tests/unit/test_settings.py` passes) |
| **Collection error (ImportError)** | **0** | — | none exist |

Total: 13 errors + (10 + 6 + 2 + 1 + 2 + 1 = 22) failures. Reconciles exactly.

### The one genuine source bug, in detail

`src/app/utils/embedding.py:5`

```python
from app.utils import logger          # binds the MODULE app.utils.logger
...
logger.warning("embedding_dimension_mismatch", ...)   # line 22 — module has no .warning
```

`src/app/utils/__init__.py:59` does `from .logger import execution_path, logger, request_state, trace_layer`,
which re-exports the loguru object. But `from .logger import ...` also causes the import system to set
`app.utils.logger = <module>` as a package attribute — and any later direct import of the submodule re-sets it,
clobbering the loguru object. `from app.utils import logger` therefore resolves to the **submodule**, which has
no `.warning`.

The fix is one line: `from app.utils.logger import logger`.

This is **not** test-only. Confirmed by running the file in isolation (`6 failed, 1 passed`) — it does not depend
on test ordering. `normalize_embedding` raises `AttributeError` on **every dimension mismatch**, and it has
**15 callers** including `shared/langgraph_layer/ingestion_kb/nodes.py`,
`shared/langgraph_layer/retrieval_kb/nodes.py`, and `features/documents/service.py`. Any embedding whose
dimension disagrees with `settings.EMBEDDING_DIMENSION` crashes the ingestion path in production. Only the
equal-dimension happy path (`test_already_correct_dim`) passes, which is why this survived.

### Why the `unwrap` bucket is stale-test, not source breakage

`src/app/features/auth/repository.py` returns `AppResult[...]` on every method (lines 52, 92, 105, 131, 157,
181, 205, 221, 242, 282, 300, 331, 359, 374, 395, 422, 440). The tests stub raw values —
`tests/integration/test_auth.py:61` `mock_repo.email_exists.return_value = False`, `:103`
`mock_repo.find_by_email.return_value = mock_user`. Production calls `.unwrap()` correctly; the mocks were
written pre-Result-migration. Corroborating: `tests/integration/test_auth.py:167` references
`mock_repo.find_by_id_result`, and `MagicMock(spec=UserRepository)` rejects it — the dual-method `*_result`
variant was collapsed out of the repository and this test was never updated.

## Blast-radius overlap

Refactor areas: ingestion · documents · tools · reconciliation · cognee · graphiti · agents · prompts.

| Failing tests | Refactor area | Disposition |
|---|---|---|
| `tests/unit/documents/test_normalize_embedding.py` (6) | **documents + ingestion** (direct hit) | **Keep and fix.** Targets `src/app/utils/embedding.py::normalize_embedding`, live on the ingestion path with 15 callers. This is a failure the refactor SHOULD fix, not delete. The single highest-value item in this baseline. |
| `tests/integration/test_search.py` (6) | **ingestion** (partial — `SearchIngestRequest`/`SearchIngestResponse`) | Keep; mocks need the `Result` update. `search` is not slated for deletion. |
| `tests/integration/test_auth.py` (10) | none | Outside blast radius. Pre-existing; leave alone. |
| `tests/integration/test_health.py` (6 err) | none | Outside blast radius. Missing `client` fixture. |
| `tests/integration/test_api_deprecation.py` (7 err) | none | Outside blast radius. Missing `client` fixture. |

**No failing test lives in a module slated for deletion.** Reconciliation, cognee, graphiti, agents, prompts,
tools, and `ingestion_kb` have **zero test coverage** — so there is nothing to delete alongside them, and
equally no test will notice if their deletion breaks something.

**Untested-but-in-scope (the real risk):** every module the refactor targets except `documents` and `search` is
uncovered. `graphify`/codegraph report "no covering tests found" for `src/app/utils/logger.py:99` and for the
reconciliation entry points. Deleting reconciliation will produce **no test signal at all** — green after that
deletion means nothing was checked, not that nothing broke.

**Delete candidate with no test attached:** `src/app/shared/rag/document_processing/todo_temp.py` does not
parse. Ruff reports `invalid-syntax` at `:406:1` (Unexpected indentation) and `:773:1` (Expected a statement),
and coverage emits `CoverageWarning: Couldn't parse Python file ... (couldnt-parse)`. A syntactically invalid
scratch file sitting inside the `document_processing` blast radius — safe and correct to delete.

## Lint / type / ast-grep baseline

### ruff — `uv run ruff check src/` → **125 errors**, exit 1

Note: this repo's ruff emits rule **names**, not codes.

| Count | Rule |
|---|---|
| 76 | `blanket-type-ignore` |
| 18 | `undefined-export` |
| 12 | `help` (hint lines, not distinct violations) |
| 8 | `import-outside-top-level` |
| 5 | `typing-only-standard-library-import` |
| 5 | `blind-except` |
| 2 | `single-item-membership-test` |
| 2 | `missing-newline-at-end-of-file` |
| 2 | **`invalid-syntax`** |
| 1 each | `unused-method-argument`, `unexpected-indentation`, `typing-only-third-party-import`, `too-many-statements-in-try-clause`, `no-self-use`, `line-contains-todo`, `f-string-in-exception` |

`No fixes available (7 hidden fixes can be enabled with the --unsafe-fixes option).`

Two standouts:

- **`invalid-syntax` ×2** — both in `src/app/shared/rag/document_processing/todo_temp.py` (`:406:1`, `:773:1`).
- **`undefined-export` ×18** — *all 18* in `src/mcp_core/__init__.py:2-19`. Its `__all__` lists 18 names that do
  not exist (`MCPClientManager`, `MCPServerHandle`, `get_mcp_server`, `serve_mcp`, …). This is precisely the
  dead/duplicate-symbol rot the refactor targets, and it explains why `tests/conftest.py:17-28` has to
  `MagicMock` out `mcp_core` and nine of its submodules — the package cannot be imported for real.

### ty — `uv run ty check src/` → **46 diagnostics**, exit 1

| Count | Code |
|---|---|
| 34 | `unresolved-attribute` |
| 7 | `unused-type-ignore-comment` |
| 2 | `invalid-assignment` |
| 1 | `unused-ignore-comment` |
| 1 | `unsupported-operator` |
| 1 | `invalid-argument-type` |

Per-file concentration — heavily inside the blast radius:

| Count | File |
|---|---|
| 11 | `src/app/shared/langchain_layer/agents/tools/precedent_tools.py` |
| 5 | `src/app/shared/langchain_layer/agents/middlewares/guardrails.py` |
| 4 | `src/app/shared/langchain_layer/agents/tools/get_obligation_chain.py` |
| 3 | `src/app/shared/langchain_layer/agents/factory.py` |
| 3 | `src/app/features/documents/service.py` |
| 3 | `src/app/features/auth/service.py` |
| 2 each | `shared/crawler/processor.py`, `features/profile/service.py`, `features/documents/graphiti_verifier.py`, `features/crawler/service.py`, `features/auth/websocket_security.py` |
| 1 each | `langgraph_layer/kb_retry.py`, `langgraph_layer/ingestion_kb/nodes.py`, `langgraph_layer/checkpointer.py`, `langchain_layer/callback.py`, `agents/tools/crawl.py`, `agents/memory/cognee_client.py`, `shared/crawler/validator.py`, `features/documents/legal_metadata.py` |

`agents/tools/` alone accounts for 15 of 46. One is a latent runtime bug of the same family as the logger one:
`src/app/shared/langgraph_layer/ingestion_kb/nodes.py:238` — `warning[unresolved-attribute]: Object of type
'dict[str, Any]' has no attribute 'doc_id'` on `f"doc_id={state.doc_id}"`, i.e. `state` is a dict but is being
attribute-accessed.

### ast-grep — `ast-grep scan src/` → **4 errors**, process exit 0

```
Error: 4 error(s) found in code.
Help: Scan succeeded and found error level diagnostics in the codebase.
```

Rules firing: `no-raw-httpexception` (incl. `src/app/examples/redis_examples.py:299`) and
`no-raise-app-error-mapper` (`src/app/features/users/service.py:48` — `raise app_error_to_exception(error)`).
Five rules are vendored at `.ast-grep/rules/` per `sgconfig.yml`.

**Caution for the verifier:** `ast-grep scan` exits **0** even with error-level diagnostics. Exit code is not a
usable gate here — you must compare the printed count against the baseline of 4.

## Service matrix

Every entry is "not reachable", and for this suite that is **irrelevant** — no test touches any of them.

| Service | Expected by | Reachable? | How to start |
|---|---|---|---|
| Postgres/Timescale :5432 | app runtime (`DATABASE_URL`); **no test** | **Not listening** | `docker compose up -d timescale` |
| Redis :6379 | app runtime (`REDIS_URL`); tests use `fakeredis` | **Not listening** | Not in `docker-compose.yml` — no service defined |
| MongoDB :27017 | app runtime (`MONGODB_URL`); tests `MagicMock` Beanie docs | **Not listening** | Not in `docker-compose.yml` |
| Neo4j :7687 | `graphiti-core`, `rag/graphiti/subgraph.py`; **no test** | **Not listening** | Not in `docker-compose.yml` |
| RabbitMQ :5672 | Celery (`RABBITMQ_URL`); tests `MagicMock` `tasks` | **Not listening** | `docker compose up -d rabbitmq` |
| S3/MinIO | `boto3`/`mypy-boto3-s3`; **no test** | n/a | Not in `docker-compose.yml` |

`docker ps` → **no containers running**. `ss -ltnp` shows only editor/terminal/DNS sockets; no database port is
bound. Note that `docker-compose.yml` defines only `rabbitmq`, `timescale`, `caddy`, `ai-service-1` — there is
**no Redis, Mongo, Neo4j, or MinIO service anywhere in the compose file**, so compose could not fully provision
this stack even if it were up.

### Test-infrastructure findings

- **No marker skips service-dependent tests.** `pyproject.toml:766-770` declares `slow`, `integration`, `unit`
  under `--strict-markers`. None is applied as a skip condition; `tests/integration/` is "integration" in name
  only — it mocks everything.
- **No testcontainers, no `.env.test`, no `docker-compose.test.yml`.** `testcontainers` and `pytest-timeout` are
  **not installed**. Installed test plugins: `pytest-asyncio` (`asyncio_mode = "auto"`), `pytest-cov`,
  `fakeredis`, `faker`, `pytest-subtests`.
- **`.env.example` exists** (services at `localhost`) and `.env.development` exists, but **no test reads either**.
- **There is no documented way to run tests WITH services, because none is needed.** `Makefile` offers only
  `make test` → `uv run pytest -x`.
- **Structural hazard, currently latent:** `tests/unit/billing/` and `tests/unit/search/` lack `__init__.py`
  while `tests/unit/` and `tests/unit/documents/` have one. Two files share the basename
  `test_circuit_breaker.py` (`tests/unit/` and `tests/unit/billing/`). Under pytest's default `prepend` import
  mode this normally raises "import file mismatch" — it is not firing today, but adding a third same-named file
  or touching the layout will trip it.

## Verifier instructions

### The trap: coverage makes a fully green suite exit 1

`pyproject.toml:752-760` puts `--cov-fail-under=80` in `addopts`. Current total coverage is **18.38%**:

```
FAIL Required test coverage of 80% not reached. Total coverage: 18.38%
```

**Exit code is unusable as a pass/fail signal for this suite.** Fixing every one of the 35 failures would still
exit 1. Compare the summary line, not `$?` — or add `--no-cov`.

### Reproduce this baseline

```bash
cd /home/harmeet/Desktop/Projects/langchain-fastapi-production
uv run pytest -q --no-header -p no:cacheprovider
```

### "No new failures" means exactly

```
22 failed, 55 passed, 13 errors
```

and the failing set is exactly the 35 IDs listed in the taxonomy above. A quick diff:

```bash
uv run pytest -q --no-header -p no:cacheprovider 2>&1 \
  | grep -E '^(FAILED|ERROR)' | sort > /tmp/after.txt
diff /tmp/baseline-failures.txt /tmp/after.txt
```

Generate `/tmp/baseline-failures.txt` from a clean tree with the same pipeline before starting work.

Interpretation rules:

- **Passed count must not drop below 55.** It should rise to 61 the moment `src/app/utils/embedding.py:5` is
  fixed — that single line converts 6 failures to passes and is the cheapest verifiable win in this baseline.
- **Any new ID in the failing set is introduced.** No exceptions, no environmental excuse: this suite cannot
  fail for environmental reasons.
- **Deleting reconciliation/cognee/graphiti modules must not change these numbers at all.** Those modules have
  zero coverage. If any number moves, something imported them that you did not expect — and that is a finding,
  not noise. Unchanged numbers there mean nothing was verified, not that nothing broke.
- Faster inner loop while iterating: `uv run pytest -q --no-header -p no:cacheprovider --no-cov` (~0.8s for a
  single file, <20s for the suite).

### The other three rungs

```bash
uv run ruff format --check src/   # NOT captured in this baseline — see Fog
uv run ruff check src/            # baseline: 125 errors, exit 1
uv run ty check src/              # baseline: 46 diagnostics, exit 1
ast-grep scan src/                # baseline: 4 errors, exit 0 (compare count, not exit code)
```

All three lint/type rungs are **red before the refactor starts**. GREEN cannot mean "exit 0" for this project
until these are driven down; it must mean "no worse than 125 / 46 / 4".

Two of the 125 ruff errors are `invalid-syntax`, so **`src/` does not fully parse today**. Deleting
`src/app/shared/rag/document_processing/todo_temp.py` drops ruff to ~122 and removes the coverage parse warning.

## Fog

- **`uv run ruff format --check src/` was never run.** The brief's task list specified `ruff check` and `ty
  check` and I ran exactly those; the format rung is genuinely unmeasured. Given 2 `invalid-syntax` errors,
  `ruff format --check` will likely fail to parse `todo_temp.py` too. The verifier should establish this rung's
  baseline before relying on it.
- **The 12 `help` entries in the ruff table are hint lines**, not distinct violations — my `grep -oP` for
  leading rule names catches ruff's `help:` prefix. Real violation count is 125 per ruff's own footer; treat the
  per-rule table as approximate at the margin.
- **`git stash` was not used** to confirm pre-existence. Nothing needed it: the working tree contains no
  modified source files (only two untracked docs, `docs/fuzzy-crafting-cookie.md` and `docs/plan-ingestion.md`),
  so this baseline **is** the clean-tree state by definition. Note this contradicts an earlier orchestrator claim
  that 4 source files were modified — as of this run, `git status` shows no tracked-file modifications.
- **Flakiness tested only at N=2**, both runs identical. Two runs cannot rule out low-probability flake, though
  a fully mocked suite with no I/O and no `freeze_time` has little room for it.
- **Coverage at 18.38% is measured against a suite that mocks its dependencies**, so it overstates how much
  behaviour is actually exercised. Lines executed under `MagicMock` collaborators are counted as covered.
- **I did not verify whether the 13 `client`-fixture errors were ever passing.** A `client` fixture may have been
  deleted from `conftest.py`, or these tests may have been written against a fixture that never landed. `git log
  -p tests/conftest.py` would settle it; I did not run it.
- **`tests/performance/todo.md` is 70 KB of numbered todos, not tests.** `tests/e2e/` and `tests/unit/shared/`
  contain no test files. Real test surface is 14 files.
