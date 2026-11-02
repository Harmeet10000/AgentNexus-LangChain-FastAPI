> Change class: **M**. **All implementation tasks are trigger-deferred** — each names the landed artifact that opens it. The only immediately doable work is section 1 (the ledger rows).

## 1. Ledger rows (do now)

- [x] 1.1 Add the two surface rows to `docs/quarantine-ledger.md` per the format: the empty `preamble` column (kept because G2 is deferred; port = fill via contextual-chunk-retrieval, discard = drop the column; trigger = the measurement baseline lands) and the reserved fourth-branch registration point (kept because G4 is deferred; port = admit the branch with a measured weight, discard = remove the reserved parameter; trigger = edges + baseline). Verify: rows follow standing rule 4 (concrete triggers); `retrieval-measurement-gate` and `mount-retrieval-graph` non-goal sections may then reference the rows instead of restating the deferral.

## 2. G2 — contextual chunk retrieval (trigger: baseline landed)

- [ ] 2.1 Amortized preamble generation at ingest (document-level context cached per document; provider calls ∝ documents, not chunks); search text includes the preamble; disabled documents explicitly marked context-free. Verify: preamble-fill + call-proportionality tests; the disabled-state distinction is observable.
- [ ] 2.2 Measured adoption: baseline comparison per document class decides default-on; regression reverts. Verify: adoption decision cites a recorded comparison; revert test green.

## 3. G4 — fourth fused branch (trigger: edges ✅ + baseline landed)

- [ ] 3.1 Implement the branch at the reserved registration point with weighted RRF fusion; admit only with a recorded weight-citing comparison; graphless corpora rank identically to three-branch behaviour. Verify: registration test, graphless-equality test, weight-cites-baseline test; participating-but-empty recorded in the decision event.

## 4. G5 + G6-remainder — loop hydration and budgets (trigger: spans slice landed; G5 also needs baseline attribution)

- [ ] 4.1 Per-stage cost/latency budgets on the loop built on the spans; exhaustion → typed degraded outcome naming the stage. Verify: over-budget degradation test.
- [ ] 4.2 Standing degradation alert, distinct from and earlier than the harden change's pilot-kill criteria; attributable per stage and document class. Verify: alert-before-kill test.
- [ ] 4.3 Section hydration behind the measured-evidence gate; citations stay clause-precise. Verify: citation-precision-under-hydration test + recorded baseline effect.

## 5. G7 — feedback capture (trigger: console signals + harness ingestion)

- [ ] 5.1 Signal capture with query, served identifiers, evidence policy; no content beyond existing audit contracts. Verify: capture-completeness test + content-boundary check.
- [ ] 5.2 Review-gated promotion into the golden set with provenance; unreviewed signals change no gate. Verify: promotion test + immutability test.

## 6. Gates and archive

- [ ] 6.1 At any implementation landing: `uv run ruff check --fix src/` · `uv run ty check src/` · `uv run pytest` · `ast-grep scan src/` — green.
- [ ] 6.2 Archive syncs the four spec deltas; resolving a ledger row happens in the same change as its implementation (standing rule 2).
