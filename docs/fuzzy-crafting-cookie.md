# Refactoring Plan — AgentNexus-LangChain-FastAPI

## Context

This repo is ~41k LOC of well-structured Python (FastAPI + LangGraph, 178 configured lint rules, `SecretStr` config discipline, a correct HITL WebSocket protocol) that **has never executed end to end**. A review found the quality problems are not primarily code smells — they are a broken toolchain, a never-wired feature, and one live production bug hiding behind a partially-initialized import.

Concretely, today:

- `uv run pytest` **cannot** exit 0 — `--cov-fail-under=80` sits in default `addopts` (`pyproject.toml:749`), so every run is red regardless of results.
- `uv run ruff check src/` and `uv run ty check src/` **cannot** pass — `src/app/shared/rag/document_processing/todo_temp.py` is 783 lines of unparseable Python (`__all__` closes at :404, an orphaned class body resumes at :406). Zero importers.
- `uv run alembic upgrade head` **fails** — two heads (`0001`, `a71f0d7d9c12`) diverge at `2bc7726317f6`.
- `src/app/utils/embedding.py:5` raises `AttributeError` on every embedding-dimension mismatch (see B1 below).
- Six mounted `/api/v1/documents/*` endpoints return 500 because their auth dependency reads `request.state.user_id`, which nothing ever sets.

So CI has four independent reasons to be red before a single test result is considered. **The intended outcome of this plan is a repo where the toolchain is trustworthy, the mounted endpoints work, and the genuinely-refactorable code is refactored behind test coverage** — in that order, because the later work is unsafe without the earlier.

Scope agreed: the **full sequence, Phases 0–6 (~7 days)**. The `client` fixture uses a **real stack, gated by the `integration` marker**.

## Ground rules

From the refactor skill, and they decide the phase ordering:

1. **Behavior is preserved** — a change that alters behavior is a *fix*, not a refactoring, and gets its own commit and its own test.
2. **Tests first** — "critical production code without tests → add tests FIRST." This is why Phase 3 precedes Phase 4.
3. **One small change per commit**, verified before the next.
4. **Never mix** a refactoring with a feature change.
5. **Delete, don't improve**, code with zero callers.

## Corrections to carry into execution

Prior analysis was wrong on these. Do not chase them:

| Claim | Reality |
|---|---|
| Alembic heads are `8a7d9b1c2e3f` + `a71f0d7d9c12` | Heads are **`0001`** and **`a71f0d7d9c12`**. `8a7d9b1c2e3f` has a child (`9f4a1b7c6d2e`). Branch point `2bc7726317f6` is correct. |
| Five duplicated auth stubs | **Four.** `src/app/features/crawler/router.py:25` is `get_client_identifier` — guards with `hasattr`, falls back to `X-Forwarded-For` then `request.client.host`. Correct defensive code; **leave it alone**. |
| conftest mocks hide a `features/` ⇄ `shared/` layering violation | **No such violation** — zero runtime imports from `shared/` into `features/`. The mocks hide import-time side effects plus one cross-root cycle. See Phase 6. |
| ~380 duplicated lines in `nodes.py` | **~130.** Ten `AgentError` blocks (~88 lines) plus four similar structured-LLM bodies. The `qna`/`planner`/`human_review`/`ingestion`/`deep_research` nodes have distinct HITL logic and are **not** duplicates. |

---

## Phase 0 — Make the toolchain trustworthy (1 day) — NON-NEGOTIABLE

Nothing later is safe until `pytest`, `ruff`, and `ty` all exit 0 on a clean checkout. Cut branch `refactor/phase-0-green`.

| Step | Change | Type | Commit |
|---|---|---|---|
| 0.1 | Delete `src/app/shared/rag/document_processing/todo_temp.py` | Delete | `chore: delete unparseable dead module todo_temp.py` |
| 0.2 | `src/app/utils/embedding.py:5` → `from app.utils.logger import logger` | **Fix (B1)** | `fix(utils): bind real logger in embedding, not the module` |
| 0.3 | Remove the four `--cov*` flags from `addopts` (`pyproject.toml:745-749`); keep coverage only in CI | **Fix** | `test: remove coverage gate from default addopts` |
| 0.4 | Add `-m "not integration"` to `addopts`; add a real-stack `client` fixture in a new `tests/integration/conftest.py` | Test infra | `test: add marker-gated integration client fixture` |
| 0.5 | CI: add a `pytest tests/ -m integration` step after the service-readiness wait; point CI at the env file `settings.py` actually loads | **Fix** | `ci: run integration suite and fix env file mismatch` |

**B1 detail — the live bug.** `src/app/utils/__init__.py:35` runs `from .embedding import normalize_embedding` *before* `:59` runs `from .logger import ... logger ...`. So when `embedding.py:5` does `from app.utils import logger`, the package is mid-initialization and has no `logger` attribute yet; Python falls back to importing the submodule and binds the **module** `app.utils.logger`. The real Logger is a name *inside* that module (`src/app/utils/logger.py:99`: `logger = loguru_logger.patch(...)`). So `logger.warning(...)` at `embedding.py:22` hits a module with no `warning`. Ruff's isort keeps `.embedding` sorted before `.logger`, so **the formatter enforces the hazardous order** — the explicit `from app.utils.logger import logger` is the only stable fix. This also removes the repo's single module-level import cycle.

**0.4 detail.** `integration` is already a registered marker (`pyproject.toml:758`) and `--strict-markers` is on, so no new config is needed. `tests/integration/test_health.py:16` and `test_api_deprecation.py:12` both request `client` through an `autouse` `_setup(self, client)` fixture — 13 tests error at setup today. Build the fixture from the app factory in `src/app/main.py` and **use `TestClient` as a context manager** so lifespan runs; `/health` reports real dependency status and is meaningless without it. Because pytest applies command-line args after `addopts` and the last `-m` wins, CI's `-m integration` cleanly overrides the default in 0.5.

**Gate:** `uv run pytest`, `uv run ruff check src/ tests/`, `uv run ruff format --check src/ tests/`, `uv run ty check src/` all exit 0. Commit. Do not proceed otherwise.

---

## Phase 1 — Delete dead code (0.5 day)

Zero-caller code needs no characterization test; the linter and type checker *are* the test. Shrinks the surface every later phase must reason about.

Targets, all in `src/app/shared/langgraph_layer/agent_saul/factory.py` and `src/app/shared/rag/graphiti/registry.py`:

- `AgentRegistry.orchestrator_agent` — declared, built with a real `create_agent` call, stored, **never read**. It costs an LLM agent construction at startup; `_build_graph_nodes` builds a fresh structured chain instead.
- `ToolRegistry.compliance_tools` / `.risk_tools` — zero callers.
- `get_current_user_id` + `CurrentUserIdAnnotated` in `src/app/features/agent_saul/dependencies.py` — zero references; the router already uses the real `CurrentClaims`.
- Stale docstrings in `registry.py` (a wrong import path, and a claim about which nodes call the memory writers). This doc rot generated three false findings during review — fixing it is cheap and prevents recurrence.

One commit per bullet. Verify `ruff` + `ty` + `pytest` after each.

**Do not delete `deep_research_results`** (`state.py`) — see the do-not-touch list.

---

## Phase 2 — Fix the live 500s (1.5 days)

A **fix**, deliberately separate from any refactoring. Four stub sites read `request.state.user_id`, which no middleware or dependency in `src/` ever sets — confirmed by exhaustive grep.

Reuse the **working** pattern already in the repo: `CurrentClaims` / `get_token_claims` from `src/app/features/auth`, applied exactly as `src/app/features/agent_saul/router.py:52` does it. Do not invent a new dependency.

| Step | Change | Commit |
|---|---|---|
| 2.1 | Write *failing* integration tests: authenticated `GET /api/v1/documents/...` asserts 200/404, never 500 | `test: cover documents endpoints require real auth` |
| 2.2 | `src/app/features/documents/dependencies.py` — swap the stub for `claims: CurrentClaims` → `claims.sub` | `fix(documents): use real token claims instead of request.state stub` |
| 2.3 | Same pattern in `search/dependencies.py` and `ingestion/dependencies.py` (unmounted, so latent) | `fix(search,ingestion): replace auth stubs with token claims` |
| 2.4 | `agent_saul/dependencies.py` — return 503 when `app.state.saul_graph` is absent, mirroring the existing `get_saul_checkpointer` guard in the same file | `fix(agent_saul): raise 503 when graph is not wired` |

**Verify:** 2.1's tests fail before 2.2 and pass after. This is the pattern-repeated change in this plan — three files, one identical edit shape.

---

## Phase 3 — Characterization tests for `nodes.py` (1 day)

Satisfies ground rule 2 before Phase 4 touches anything. All targets are pure functions or stub-injectable — **no LLM, no DB, no network**. File: `src/app/shared/langgraph_layer/agent_saul/nodes.py`.

| Step | Target | Commit |
|---|---|---|
| 3.1 | `route_after_qna`, `route_from_orchestrator`, `route_deep_research` — table-driven across every `WorkflowStatus` / `OrchestratorActionType` | `test: characterize agent_saul routing functions` |
| 3.2 | `dispatch_entity_extraction` — one `Send` per segment; assert the `working_memory` defaults | `test: characterize entity-extraction fan-out` |
| 3.3 | `_build_analysis_context` — exact `CLAUSES:/ENTITIES:/RELATIONSHIPS:` layout and the clause truncation | `test: characterize analysis-context builder` |
| 3.4 | **Every node's failure branch** via a stub `Runnable` — assert the exact `{"status": ..., "errors": [AgentError(...)]}` payload at all ten sites | `test: characterize agent_saul node error payloads` |

3.4 is the gate for R1. **If a test needs a production change to pass, that is a bug** — route it to Phase 2's column, do not "fix" it here.

---

## Phase 4 — The actual refactorings (1.5 days)

878 → ~750 lines in `nodes.py`. Order matters: R2 before R1 (fewer sites to convert), R1 before R3 (R3 reuses the helper).

| Step | Item | Operation | Smell |
|---|---|---|---|
| 4.1 | R2 — collapse the two consecutive byte-identical `MISSING_NORMALIZED_DOCUMENT` guards into one `is None` check | Consolidate Conditional | Duplicated Code |
| 4.2 | R1 — extract `_fail()`; convert all ten sites, **one node per commit** | Extract Method | Duplicated Code |
| 4.3 | R7 — move `_utc_now_iso` into the module's helper block | Move Method | Poor cohesion |
| 4.4 | R6 — name the slice literals (`_HUMAN_REVIEW_SEGMENT_PREVIEW`, `_REFLECTION_LOG_CHARS`, `_CLAUSE_CONTEXT_CHARS`) | Replace Magic Number with Constant | Magic Numbers |
| 4.5 | R5 — rename `_extract_risk_output` → `_placeholder_risk_output` (and the compliance twin) | Rename | Misleading Name |
| 4.6 | R3 — extract `_invoke_structured()` for the four similar structured-LLM nodes | Extract Method | Duplicated Code |

**R1 constraint that must not be missed:** `_fail()` takes `status` as a **parameter** defaulting to `FAILED`. Two of the ten sites use a different status (`PLAN_REJECTED`, `COMPLETED`). Collapsing those to `FAILED` silently changes behavior.

**R2 equivalence note:** the second guard exists only to narrow the type for `ty` (`if not state.get(...)` does not narrow). The original also failed on falsy-but-not-`None`; Pydantic models are always truthy, so `is None` is equivalent.

**R3 stopping condition:** keep the four call sites as thin closures over one helper. **If the helper needs more than 4 parameters, abandon R3** — a Long Parameter List is worse than the duplication it removes.

**R5 rationale:** renaming is the whole change. These two functions discard the LLM result and return hardcoded `LOW` risk / `compliant`. Implementing them properly requires threading `ToolRegistry` into `build_agent_registry`, which does not accept it — that is **feature work** (see Deferred), and mixing it in violates ground rule 4. The rename makes the placeholder self-documenting until then.

**Verify:** `pytest` green after *every* commit; `git diff --stat` shows a net line decrease.

---

## Phase 5 — Node-name single source of truth (0.5 day)

Deliberately split into one refactoring and two fixes so the behavior change is isolated and reviewable. Files: `graph.py`, `state.py`, `nodes.py` in `src/app/shared/langgraph_layer/agent_saul/`.

- **5.1 (refactor)** Introduce `GRAPH_NODE_NAMES` as the single source; derive `graph.py`'s `add_node` loop from it. Safe because `state.py` and `graph.py` already list the identical 16 names. Redefine `_VALID_WORKER_NODES` as an explicit **subset** of it so the two cannot drift.
- **5.2 (fix)** The orchestrator `path_map` lists 5 destinations but `route_from_orchestrator` can return any of 10 worker names — six are unreachable and a valid routing decision raises at runtime. Expand the `path_map`.
- **5.3 (fix)** `graph.py` fans `relationship_mapping` out to `risk_analysis` **and** `compliance` in one superstep; both write `status`, which has **no reducer** in `state.py`. LangGraph raises `InvalidUpdateError: can receive only one value per step`. Either give `status` a reducer or serialize the two nodes. This is strong evidence the graph has never run end to end.

**Verify:** a compile-only test — call `build_saul_graph(...)` with stub LLMs and `checkpointer=None`, assert `.compile()` succeeds and `get_graph().nodes` has 16 entries. This is the first test that ever exercises `build_saul_graph`.

---

## Phase 6 — Dependency direction and config hygiene (1 day)

The deepest item, and the one that finally retires the conftest mocks.

**What the import graph actually shows** (all 307 modules, runtime edges only, `TYPE_CHECKING` excluded):

- `shared/` → `features/`: **zero** imports. The macro layering is clean; there is no inversion to fix.
- Direction evidence is decisive: `tasks` / `mcp_core` / `database` → `app` is **49 imports across 23 files**; `app` → those roots is **exactly 1**. `app` is the core; the others are outer layers. That single inversion is `src/app/connections/__init__.py:3` (`from mcp_core.mcp import ...`), and `mcp_core/client/manager.py` imports back into `app.config` and `app.utils` — a hard `app` ⇄ `mcp_core` cycle at package granularity.
- `app.connections` ⇄ `app.shared` is **bidirectional**. `connections` is the lower layer (raw clients) and must not import `shared`. Two edges to invert — notably `connections/crawl4ai.py` → `shared.crawler` while `shared/.../tools/crawl.py` → `connections.crawl4ai`.
- **What the mocks really hide:** import-time side effects. `connections/celery.py` calls `get_settings()`, builds `celery_app`, and creates OTel meters *at module scope*, and `connections/__init__.py` imports it — so merely importing `app.connections` constructs a Celery app. Plus a Beanie `Document` needing live Mongo, and two heavy `langgraph_layer` subtrees.

| Step | Change | Type |
|---|---|---|
| 6.1 | Fix two stale `pyproject.toml` paths: `version_locations` points at a nonexistent dir and conflicts with `alembic.ini:10`; per-file-ignores target a nonexistent `src/alembic/migrations/` | Fix |
| 6.2 | `alembic merge` heads `0001` and `a71f0d7d9c12`; confirm `upgrade head` reaches the `documents`/`chunks` tables | Fix |
| 6.3 | Invert `app` → `mcp_core`: declare the MCP-manager protocol in `app/connections/`, have `mcp_core` supply the implementation, inject at lifespan — the pattern `lifespan.py` already uses for Celery | Refactor |
| 6.4 | Make `connections/celery.py` import-side-effect-free; defer construction to lifespan | Refactor |
| 6.5 | Delete each now-unnecessary `sys.modules[...] = MagicMock()` line in `tests/conftest.py:11-19` and **correct the misleading comment at line 10** | Cleanup |

6.4 alone removes four of the nine mocks. `sys.modules["app.connections.mcp"]` (line 11) appears vestigial — no module under `src/app/` imports it; confirm and drop.

**Invert `connections` → `shared`** as part of 6.3/6.4: move the crawler config down into `connections/`, and move OTel setup out of `connections/celery.py` into `lifecycle/`.

---

## Do NOT touch

| Item | Decision | Why |
|---|---|---|
| `src/lynk/` (Go, ~1254 LOC + real tests) | **Leave entirely alone** | It works and won't change again — the skill's first "when NOT to refactor" case. Own `go.mod`, hexagonal layering, golden testdata. One caveat: it sits under `src/`, which `pyproject.toml` package-discovery scans and `--cov=src` measures. Relocating to repo-root `lynk/` is a `git mv`, not a refactor — defer. |
| `crawler/router.py:25-34` | Leave alone | Not a stub. Correct defensive code. |
| `factory.py` `_build_graph_nodes` Long Parameter List | Leave alone | Textbook Introduce Parameter Object, but its only caller is `build_saul_graph`, which has no real callers. Refactoring it is "just because." Revisit only if the graph gets wired. |
| `deep_research_results` (`state.py`) | **Do not delete** | Nothing reads it, but it is the entire output of the `deep_research` node — one of the few nodes with real logic. Deleting it silently deletes unfinished work. |
| `write_clause_episodes_to_graphiti`, `write_final_report_to_memory` | Leave alone | Zero callers, but they are *implemented* integration points, not accidents. Deleting destroys work; calling them is feature work. Fix the docstring (Phase 1) and stop. |
| The five unmounted routers | Neither mount nor delete | Mounting drags in Phase 2's auth work ×5 plus untested services. Deleting throws away real features. Instead **add a test asserting the mounted set in `src/app/api/v1.py` is exactly what you intend**, so the drift is explicit rather than invisible. |
| The two `_placeholder_*` bodies | Rename only | Implementing them is feature work requiring a signature change across `graph.py`/`factory.py`. |

---

## Verification

**Per commit** (every single one):
```
uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/
uv run ty check src/
uv run pytest
```

**Per phase gate:**
- **Phase 0** — all four commands above exit 0 on a clean checkout. This is the gate for the entire plan.
- **Phase 2** — with the stack up: `uv run pytest tests/ -m integration`. The `documents` tests must go from 500 to 200/404. Manually confirm with a real token against a running app (`uv run uvicorn src.app.main:app --reload --reload-dir src --port 5000`).
- **Phase 3** — new tests green against *unmodified* production code.
- **Phase 4** — `pytest` green after each commit, and `git diff --stat` shows a net decrease.
- **Phase 5** — the new compile-only test asserts `build_saul_graph(...).compile()` succeeds with 16 nodes.
- **Phase 6** — `uv run alembic upgrade head` exits 0 and reaches the `documents`/`chunks` tables. Then delete a conftest mock and confirm `pytest` still passes; that is the proof the cycle is genuinely gone rather than relocated.

**End-to-end, once Phase 6 lands:** push the branch and confirm the full `.github/workflows/test.yml` job is green — migrations, ruff, format, `ty`, unit tests, integration tests, coverage upload. That has apparently never happened in this repo, and it is the real acceptance criterion.

## Deferred (feature work — explicitly out of scope)

Real risk/compliance extraction (requires threading `ToolRegistry` into `build_agent_registry`); mounting the five dormant routers; uncommenting the checkpointer and `saul_graph` wiring in `lifespan.py` (blocked on 5.2 and 5.3); calling the two memory writers; integrating `src/lynk/` into the Python surface or CI. Also unaddressed here: the committed `db_password = "change-me-..."` in all three `infra/gcp/terraform/environments/*.tfvars` alongside `enable_public_access = true`, and the `CORS_ORIGINS` default of `["*"]` — both are security items, not refactorings, and warrant their own pass.
