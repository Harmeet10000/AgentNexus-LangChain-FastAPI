# Review — ingestion-truthfulness-residue

## Coverage check: residues → requirements

| Residue (retired plan / audit) | Where it lands |
|---|---|
| Phase E: `docs/quarantine-ledger.md` with retroactive rows + standing rule | `quarantine-ledger` — all three requirements |
| Phase E: state key-set seatbelt test | `ingestion-state-boundary` — requirement 1 |
| Phase E: gate the ungated silent-drop edge (`legacy_service.py:58-67` vs `workflow.py:49-67`) | `ingestion-state-boundary` — requirement 2 |
| Phase C residue: cross-contract leakage assertion | `hierarchical-document-chunking` ADDED, scenario 1 |
| Phase C residue: multi-contract classification-boundary suite (C5) | `hierarchical-document-chunking` ADDED, scenario 2 (+3) |

## Decisions to defend in review

1. **Mapping over failing** for the legacy path (design Decision 2) — reviewer should challenge whether any of the six dropped keys deserve a declared channel instead of config routing.
2. **Tests over runtime guards** for the seatbelt (Decision 3) — the parametrised suite must enumerate its covered graphs; check that enumeration mechanism in the tasks.
3. **Suite may record a red outcome** on today's regex classifier (Decision 4) — confirm the project's expected-failure conventions cover this before implementation.

## Cross-change interactions

- `mount-retrieval-graph` owns the retrieval graph's fate (cutover → the ledger's survivor row resolves); this change only records it in the ledger. The inline deletion there (its task 5.1) must update the ledger row in the same change, per the ledger's discard rule.
- `re-ingestion-readiness` gates the LangExtract batch processor's revisit trigger ("first bulk re-index"); the ledger row points at that change, not this one.
- `retrieval-measurement-gate` is independent; no shared requirements.

## Sign-off

- [ ] All residue rows above have a spec requirement
- [ ] No implementation of the port-or-discard criteria smuggled in
- [ ] Verify current line numbers cited in the proposal/design at implementation time (tree moves)
