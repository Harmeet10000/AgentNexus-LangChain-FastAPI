> Change class: **L**. Artifact review, before implementation.

## Completeness

- [x] Four requirements have concrete scenarios, including refusal, repeated execution, failed provisioning and shutdown.
- [x] Message names, queue topology, scheduler signature and operator commands are covered.
- [x] New runtime-isolation capability uses ADDED; existing outbox/deployment requirements are preserved.

## Correctness

- [x] Producer isolation is observable in a fresh process, independent of mocks.
- [x] Loop identity and close ordering can be asserted without live databases.
- [x] Registration compares filesystem discovery, payload registry and actual worker tasks.

## Standards

- [x] Existing Result/error unions and HTTP response contracts remain intact.
- [x] No new secret representation, global live client, or database naming scheme is introduced.
- [x] Typed resource objects and explicit async construction preserve project conventions.

## Risk

- [x] Internal import path break is explicit; operator commands change atomically.
- [x] Existing mock isolation fixtures must be migrated with source imports.
- [x] Shutdown cannot be guaranteed after SIGKILL; design claims orderly cleanup only.
- [x] Prefork is the supported worker model; concurrent thread use of one Runner is not supported.
- [x] Redis leases still cannot prove completion of external effects. No new exactly-once claim is made.
- [x] Keep the existing CPU dependency constraints; removing torch alone would not produce a slim API image.

## Verification Matrix

| Contract | Evidence required |
|---|---|
| Producer isolation | Import-linter and subprocess import/validation test |
| Loop ownership | Two operations, child reset, failure cleanup, repeated shutdown tests |
| Delivery compatibility | Existing topology/typed dispatch/retry tests and all registered names |
| Packaging | Lock sync, real callable entry points, Make/Compose/documentation agreement |
| Regression | Ruff, ty, pytest, ast-grep; compare failures with captured baseline |

## Verdict

VERDICT: APPROVED
