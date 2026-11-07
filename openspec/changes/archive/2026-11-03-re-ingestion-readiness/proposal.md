> Change class (pick one): **S** single-file fix / config / bump / docs · **M** feature in one module · **L** cross-cutting (multi-module, migration, security, public API)

**M** — a readiness contract for a capability that does not exist yet: the spec defines what must be true before re-ingestion of an existing document under a new pipeline version is ever exposed, without implementing the exposure.

## Why

Re-ingestion stays deliberately unexposed (retired plan, "Deferred — re-ingestion exposure", decided 2026-10-28) because three preconditions never landed: what re-ingestion means for existing Graphiti edges (replace, version, or supersede — the idempotent writer never decided), a retention policy for superseded chunks that open evidence handles and past citations still point at, and a stability contract for chunk identity (`chunk_id` surviving a re-run, or every historical `EvidenceRef` dangles). The LangExtract batch processor's quarantine revisit trigger ("first bulk re-index") is the same event that would unlock this item, so the readiness contract must exist in writing before any trigger fires.

## What Changes

- **A written readiness contract, not an implementation.** A new `re-ingestion-readiness` capability specifies, as testable requirements: re-ingestion remains unexposed (no endpoint, no admin route, never a model-callable tool) until every requirement in the capability is satisfied by its own change; Graphiti edge semantics under re-ingestion are decided and enforced by the idempotent writer (no duplicate, orphaned, or silently-stale edges); superseded chunk versions are retained while any evidence handle or recorded citation can still resolve against them; chunk identity under a re-run is either deterministic (same content + pipeline version ⇒ same `chunk_id`) or the re-ingestion re-maps evidence handles by explicit rule.
- **Preconditions 1–2 stay owned where they already are.** `pipeline_version` persistence and version-aware cache/idempotency keys are the harden change's C9 scope; this capability references them as gating inputs rather than re-speccing them.
- **When the contract is satisfied, exposure is approval-gated.** A future change that exposes re-ingestion must land it as an administrative action (operator role, audited), never a general endpoint and never an agent capability.

## Scope / Non-Goals

In scope: the readiness contract (the three undecided preconditions plus the exposure rule) as a capability spec.

Out of scope (explicit): implementing re-ingestion in any form; the pipeline-version persistence and version-aware cache keys (harden change C9); the bulk re-processing tooling (the quarantined LangExtract batch processor — its ledger row's revisit trigger is what activates this contract); the graph branch registration (`retrieval-measurement-gate`); any migration or data change — the database is not touched by this change.

## Capabilities

### New Capabilities

- `re-ingestion-readiness`: the standing gate on re-ingestion exposure — unexposed until satisfied, Graphiti edge supersede/version semantics decided and enforced, superseded-chunk retention policy defined, chunk-identity stability contract specified, and exposure (when it comes) approval-gated and audited.

## Impact

Affected code at implementation time: none beyond tests and the spec itself — this change records the contract; the requirements are verified by inspection tests (e.g., no re-ingestion route/tool exists) rather than new behaviour. The future change that satisfies the contract will touch `src/app/retrieval/graphiti/write_clause_episodes.py` (edge semantics), the chunk store schema (retention/stability), and settings. Normative input: `openspec/changes/archive/2026-10-31-plan-bulletproof-ingestion.md` (RETIRED — the "Deferred — re-ingestion exposure" section is this change's source text, preconditions 3–5 verbatim in intent); `openspec/specs/graph-entity-canonicalisation/spec.md` (canonical identity as the idempotency key — the edge-semantics requirement builds on it); harden change C9 (preconditions 1–2).
