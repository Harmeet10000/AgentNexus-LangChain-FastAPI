> Change class: **M**. This change implements no re-ingestion capability — tasks land the contract's verification only.

## 1. Inspection tests — the exposure gate is real today

- [ ] 1.1 Add an inspection test asserting no re-ingestion surface exists: `rg`-equivalent route-table scan (no re-ingest/reprocess endpoint under any API version) and a toolset/registry scan (no model-callable re-ingestion tool in any role's toolset or the agent tool registry); verify both scans fail loudly if a surface appears, turning the spec's requirement 1 into a CI-enforced gate.
- [ ] 1.2 Add the deterministic-identity probe where the current pipeline already supports it: processing the same document content twice under the same pipeline version yields identical chunk identifiers (unit-level, against the chunking path's identity derivation per `legal-document-chunking`'s version-scoped identity); verify the probe passes today or records the gap as an expected failure naming the derivation step that is non-deterministic.

## 2. Wire the contract into the standing records

- [ ] 2.1 Point the quarantine ledger's LangExtract batch-processor revisit trigger at this capability (`ingestion-truthfulness-residue` task 1.3 creates the ledger; this task completes the linkage once both land); verify a reader of the ledger row reaches this change's spec as the gate.
- [ ] 2.2 Record in the change's own notes (or issue #107 if `gh` is authenticated) that preconditions 1–2 remain owned by the harden change's C9 and that exposure additionally requires the audit trail/operator-role design; verify the note names the owning change explicitly.
