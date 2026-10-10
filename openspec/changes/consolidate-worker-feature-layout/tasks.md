> Change class: **L**.

## 1. Producer and worker separation

- [x] 1.1 Extract shared task names, payload contracts and typed registry; verify malformed/unknown dispatches fail without consumer imports.
- [x] 1.2 Separate producer configuration/topology from worker application, schedules, task bases and observability; verify all existing wire names and routes.

## 2. Feature ownership and runtime

- [x] 2.1 Relocate all six task modules to owning features, extract billing/credit jobs, and migrate references; verify explicit registration covers every task module.
- [x] 2.2 Add persistent generic worker runtime and reliability client cleanup; verify repeated operations share a loop and process reset cannot reuse inherited state.
- [x] 2.3 Move document/crawler lifecycle into worker with lazy provisioning; verify initialization is fast, first-use provisioning and cleanup behavior.

## 3. Packaging and enforcement

- [x] 3.1 Correct dependencies, first-party imports and runnable entry points; sync lockfile and verify worker/beat command loading.
- [x] 3.2 Add import-linter contracts and CI/Make checks; verify producer and feature task boundaries without blanket exemptions.
- [x] 3.3 Update Compose, README, worker documentation and affected developer references; verify operator commands agree. (Compose/README/Makefile verified agreeing via `test_documented_worker_command`; fixed stale `app.connections.celery:celery_app` refs in `docs-site/deployment/index.mdx` and `docs-site/get-started/quickstart.mdx`, added missing crawler worker + beat rows.)

## 4. Verification

- [x] 4.1 Update regression tests and add real-process isolation/registration/runtime coverage; run targeted worker, outbox and lifecycle tests. (Removed ~130 lines of pasted review prose that broke `test_runtime_isolation.py` collection; suites green: `tests/unit/celery` 77 passed, outbox 4 passed, `tests/unit/lifecycle` 22 passed.)
- [x] 4.2 Run full Ruff, ty, pytest, ast-grep, OpenSpec validation and index sync; record results, baseline differences and operational limitations in verification.md. (Done; results in `verification.md`. Fixed 2 `ty` regressions in new worker files; remaining ruff/pytest/ty/ast-grep reds are pre-existing in untouched files or environment limits.)
