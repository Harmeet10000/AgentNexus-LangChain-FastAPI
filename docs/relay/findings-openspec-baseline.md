# Findings — the openspec baseline, and what the 6 failures actually are

Established by the orchestrator on 2026-08-17 by running `openspec validate <spec-id>` per failing item.
`openspec validate --all` reports *which* items fail; it does not say why. The per-item invocation does, and the
answers change two plans.

---

## §1 — The 6 pre-existing failures have **two** causes, and neither is fixable by a change delta

| Item | Error | Cause |
|---|---|---|
| `spec/transactional-outbox` | 6 × `Requirement "<name>" must contain SHALL or MUST` | every requirement body is missing a normative keyword |
| `spec/cognee-v1-api` | `Spec must have a Purpose section` | no `## Purpose` header at all |
| `spec/noqa-documentation` | same | same |
| `spec/pattern-matching-standard` | same | same |
| `spec/typed-exception-handling` | same | same |
| `change/mintlify-documentation` | not probed | a change, not a spec; left in the baseline |

**This is the important consequence:** a change's `specs/**` deltas add and modify **requirements**. Archiving a
change applies those requirement deltas to the deployed spec. **Nothing in the delta mechanism writes a
`## Purpose` header.** So four of the six failures cannot be repaired by authoring or archiving *any* change —
they need a direct edit to the file under `openspec/specs/`, which is housekeeping outside the change flow.

This **retroactively validates D12's decision** to accept "no new failures beyond these 6" rather than "validate
`--all` passes". The baseline is not laziness; four of the six are structurally out of reach of the change
mechanism.

### Correction to `plan-change4.md`'s C1 and its claimed baseline movement

Change 4's plan states that its step 2 fixes `cognee-v1-api` — by issuing a `MODIFIED` delta removing the
redundant `improve()` mandate — and that this moves the baseline to **17 passed / 5 failed**.

**It does not.** `cognee-v1-api` fails for exactly one reason: **a missing `## Purpose` section**. The redundant
`improve()` requirement is a *correctness* defect in the spec's content, not the cause of its validation failure.
Fixing it is still right on the merits — `remember()` already runs `add()` → `cognify()` → `improve()`
(`remember.py:915-944`), so the spec mandates redundant work — but it moves nothing in the validation counts.

**The baseline stays 16 / 6 through change 4.** Any plan step, task, or Proof asserting 17/5 is wrong and must be
restated. If the Purpose header is wanted too, that is a separate one-line file edit, and it should be recorded as
such rather than smuggled into a delta that cannot carry it.

## §2 — Change 0's outbox work is **conformance to an accepted spec**, not a new capability

`openspec/specs/transactional-outbox/spec.md` is deployed and accepted, and it already says:

```
### Requirement: Outbox Table Schema
#### Scenario: Table exists
- **WHEN** migration runs
- **THEN** `outbox_events` and `dead_letter_events` tables exist

### Requirement: Migration
#### Scenario: Migration runs
- **WHEN** alembic upgrade is run
- **THEN** both tables are created idempotently
```

Both are **violated in production right now** (`findings-database.md` §8): the tables do not exist, and
`alembic upgrade head` is a no-op because `0001` is already stamped. So this is not an undocumented bug — it is an
**accepted requirement that the deployed system fails**, which is a materially stronger thing to write in a
proposal than "we should add tables".

Two authoring consequences:

1. **Use `MODIFIED`, not `ADDED`.** The requirements already exist. Inventing parallel outbox requirements in a
   new capability would create two specs for one behaviour. The `MODIFIED` copies must add the missing SHALL/MUST
   (which is what makes this spec red), and should add the failure-mode requirement the spec currently lacks
   entirely: what the relay does when the tables are absent. Today the answer is "swallow it in a catch-all and
   log a warning", which no spec sanctions.
2. **The spec's own word "idempotently" prescribes the fix**: `CREATE TABLE IF NOT EXISTS`. That is exactly the
   right shape given the stamped-but-unapplied chain, and it means the new migration under D14 satisfies existing
   wording rather than needing new wording.

## §3 — `outbox-helper-extraction` is **stale**: the spec describes code that no longer exists

The deployed spec requires:

```
- **THEN** it SHALL call `create_async_engine(get_database_url())`
- **AND** create an `AsyncSession` bound to the engine
- **AND** call `engine.dispose()` in a `finally` block
```

The actual `auth/service.py:481-500` does none of that — it uses `self._session_factory` with no engine creation
and no `dispose()`. The code was refactored (correctly: reusing the app's session factory beats building an engine
per password-reset request) and the spec was never updated.

Note this passes validation, because validation checks structure, not truth. **A green `validate --all` says
nothing about whether a spec still describes the code**, which is worth stating plainly somewhere a future
maintainer will read it.

Change 0 is already in this file for the outbox repair, so it should carry a `MODIFIED` delta bringing this spec
back in line with the session-factory implementation.

## §4 — Three existing capabilities sit in change 0's blast radius

Change 0 should extend these rather than invent outbox capability names:

| Capability | Bearing on change 0 |
|---|---|
| `transactional-outbox` | table existence + migration idempotency (§2); currently red for missing SHALL |
| `outbox-helper-extraction` | stale engine-lifecycle requirement (§3) |
| `session-required` | governs `OutboxRelay` internals. Its `run_startup_scan` requirement (*"NOT create its own `create_async_engine` separately"*) is **satisfied** by current code. Its third requirement — *"`OutboxRelay.shutdown()` and `_running` SHALL be removed"* — needs verifying against `relay.py`; if either still exists, that is an unmet accepted requirement and belongs in change 0. |

Also observed: `transactional-outbox`, `outbox-helper-extraction`, and `session-required` all carry
`## Purpose` bodies reading *"TBD - created by archiving change <x>. Update Purpose after archive."* — the archive
flow stubs Purpose and nobody ever returns. That is the same mechanism that left four specs with **no** Purpose at
all, and it is why §1's failures cluster the way they do.
