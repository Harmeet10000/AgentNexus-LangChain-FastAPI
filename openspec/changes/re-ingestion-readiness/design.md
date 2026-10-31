> Change class: **M**. The proposal covers *why* and *what*; this covers *how*. Reference the proposal — do not restate it.

## Context

The retired plan's "Deferred — re-ingestion exposure" (decided 2026-10-28) lists five preconditions:

1. `pipeline_version` persisted on chunks and surfaced in provenance — owned by the harden change (C9).
2. Version-aware cache and idempotency keys — owned by the harden change (C9).
3. Graphiti replacement semantics — undecided: the Phase-D idempotent writer made replayed writes safe but never decided replace/version/supersede.
4. Old-version retention — undecided: superseded chunks would be deleted (or not) as a side effect, with open evidence handles and recorded citations possibly pointing at them.
5. Stable evidence identities — undecided: nothing states whether `chunk_id` survives a re-run.

The same event that would create the need (a first bulk re-index) is the batch processor's ledger revisit trigger, so the contract must predate the trigger. Verified on disk (2026-10-31): the writer is `src/app/retrieval/graphiti/write_clause_episodes.py`; chunk identity is `legal-document-chunking`'s "Chunk identity is scoped to a document version"; `graph-entity-canonicalisation` already makes the canonical entity identity the idempotency key for replayed graph writes — the edge-semantics requirement extends that discipline from nodes to edges.

## Goals / Non-Goals

**Goals:**

- The three undecided preconditions become written, testable requirements before any trigger can fire.
- The exposure rule (approval-gated admin action; never an agent tool) is enforceable by inspection.

**Non-Goals:**

- No implementation of re-ingestion, no migration, no schema change, no writer change — this change is the contract, not the capability.
- No re-speccing of preconditions 1–2 (harden change C9 owns them; the exposure requirement requires them transitively by demanding every readiness requirement be satisfied).
- No decision here on *which* of replace/version/supersede the future implementation picks — the requirement is that exactly one is decided, named, and enforced.

## Decisions

### 1. A contract-only change, verified by inspection tests

The capability's requirements are written so a future change satisfies them; the only tests this change lands are inspection tests (no re-ingestion route/tool exists; identical-input chunk processing reproduces identifiers where the current pipeline already does). Spec-first was chosen over bundling with a future implementation because the plan's discipline is that the decision record must predate the trigger event, and a bundled change would let the trigger arrive first.

### 2. Edge semantics bind to the canonical-identity discipline, not to a new key space

Whatever the future policy is, edges continue to key on the canonical entity/clause identities from `graph-entity-canonicalisation`; the supersede/version decision rides on those keys rather than minting fresh per-run identities. This is what makes "no duplicate edges" testable rather than aspirational.

### 3. Retention is stated as "resolvable until released", not as a duration

A concrete TTL here would be folklore of the same kind the plan exists to kill. The requirement names the release events (handle closed, policy releases it) and leaves the duration to the implementing change, which must declare it.

### 4. Chunk-identity determinism is scoped to content + pipeline version

Same content + same version ⇒ same `chunk_id` is the stability contract; a version change is the only legitimate churn. This matches `legal-document-chunking`'s existing version-scoped identity rather than inventing a second identity scheme.

## Risks / Trade-offs

- [A contract without an implementation can rot unread] → The exposure requirement makes every future re-ingestion change cite this capability, and the ledger's batch-processor row points here; the archive path of the retired plan records the linkage too.
- [The inspection tests are weak evidence] → Accepted: the strong requirements bind the *future* change, which must land the real tests; weak evidence today is strictly better than no written contract when a bulk re-index becomes tempting.
- [Deciding nothing about replace/version/supersede feels incomplete] → Deliberate: the plan's constraint was that the writer "decides" — this change forces the decision to exist and be named, not pre-empts it.

## Migration Plan

1. Land the capability spec and the inspection tests.
2. Point the quarantine ledger's batch-processor revisit trigger at this capability.
3. Done — the next change in this space is the one that satisfies (or implements against) the contract.

## Open Questions

Which of replace/version/supersede the future edge policy picks — deliberately open, to be decided with data by the implementing change.
