# Review — retrieval-measurement-gate

## Coverage check: residues → requirements

| Residue (retired plan / audit) | Where it lands |
|---|---|
| G1: grow golden corpus 105 → 150–300, documented matrix | `retrieval-evaluation` ADDED, requirement 1 |
| G1: recall/nDCG CI gate (CI today provenance-only, `test.yml:95`) | `retrieval-evaluation` ADDED, requirement 2 |
| G4: Graphiti as 4th fused RRF branch (registration point `documents.py:~120-129`) | `graph-fusion-branch` — all three requirements |
| G4: measured weight, default-off until measured | `graph-fusion-branch` requirement 2 |
| G5/G6/G7 — untouched, tracked on issue #107 | Explicit non-goals; no requirement |

## Decisions to defend in review

1. **Offline CI gate** (Decision 1) — reviewer should confirm the recorded-artefact refresh story: who regenerates them, how often, and that staleness is visible in the report metadata.
2. **Cutover-agnostic registration** (Decision 3) — verify against `mount-retrieval-graph`'s parity work that the branch registry contract really survives the inline deletion unchanged.
3. **Threshold set by baseline report, not invented** (Migration step 2) — confirm the report lands in `evals/reports/` and is cited from the workflow file.

## Cross-change interactions

- `mount-retrieval-graph`: reserves the branch-#4 registration point and excludes G4 explicitly; this change fills it. Sequencing: either order works — the registry contract is what both bind to.
- `ingestion-truthfulness-residue`: SRC-04 multi-contract corpus facts shared read-only; no requirement overlap.
- `re-ingestion-readiness`: independent; the graph branch reads existing edges and does not gate on re-ingestion policy.

## Sign-off

- [ ] Every G1/G4 residue row above has a spec requirement
- [ ] No default-on flip for the graph branch smuggled in
- [ ] Corpus growth tasks preserve the provenance validator's contract
