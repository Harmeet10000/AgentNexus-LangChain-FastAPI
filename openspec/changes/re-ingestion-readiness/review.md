# Review — re-ingestion-readiness

## Coverage check: residues → requirements

| Residue (retired plan "Deferred — re-ingestion exposure") | Where it lands |
|---|---|
| Preconditions 1–2 (`pipeline_version` provenance; version-aware cache/idempotency keys) | Owned by harden change C9 — referenced as gating inputs, not re-specced |
| Precondition 3: Graphiti replace/version/supersede semantics | Requirement 2 |
| Precondition 4: old-version chunk retention policy | Requirement 3 |
| Precondition 5: `chunk_id`-survives-rerun stability contract | Requirement 4 |
| Exposure rule: unexposed; approval-gated admin action when it lands; never an agent tool | Requirement 1 |

## Scoping decision (recorded per the propose skill's rules)

R3 ships as its own coherent change rather than a deferred section inside R1/R2: the `re-ingestion-readiness` capability is a complete, self-contained contract whose requirements a future change satisfies wholesale — folding it into `ingestion-truthfulness-residue` (whose scope is tests + a ledger + one edge fix) would leave the contract half inside a change that never touches the graph writer, and the skills' "one spec file per capability, complete capability specs" rule favors the standalone change.

## Decisions to defend in review

1. **Contract-only, inspection-tested** (Decision 1) — reviewer should confirm the inspection tests are worth their CI cost.
2. **No pre-choice of replace/version/supersede** (Decision 2, Open Questions) — confirm reviewers agree the decision belongs to the implementing change with data.
3. **Retention stated as release events, not a TTL** (Decision 3) — confirm this does not clash with any existing data-retention policy in settings.

## Cross-change interactions

- `ingestion-truthfulness-residue`: its quarantine ledger points the batch-processor row's revisit trigger at this capability (its task 1.3).
- `retrieval-measurement-gate`: independent — the graph fusion branch reads existing edges and never re-ingests.
- harden change C9: preconditions 1–2 are its tasks; this capability's requirement 1 makes them transitive gates on exposure.

## Sign-off

- [ ] Preconditions 3–5 each have a requirement
- [ ] No exposure mechanism smuggled in (no endpoint, tool, or flag specced)
- [ ] Preconditions 1–2 not duplicated
