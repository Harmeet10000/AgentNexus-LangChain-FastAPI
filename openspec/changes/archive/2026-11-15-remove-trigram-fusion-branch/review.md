> Change class: **S** (short review). Role: reviewer, not author.

## Verdict

APPROVED for archive. Owner directive + E4 evidence; both doors agree on the
new two-branch contract; the normalization drift the removal unmasked is fixed
in the same change (it was a real two-door divergence, not cosmetic).

## Checks that matter

- [x] No spec contradiction: no synced spec pins trigram by name (verified by
  search over openspec/specs/); the delta adds the composition contract to
  hybrid-retrieval-ranking rather than amending a removed requirement.
- [x] The offline harness keeps its trigram sweep (the measurement that
  justified the removal survives its own decision).
- [x] repo.trigram_search retained, unfused, docstring-cited.
- [x] Parity re-record: only ranking-order expectations changed; shape,
  provenance, fallback, cache assertions untouched.

## One follow-up owed

The harness's `--branches` documentation strings still list trigram as a
*served* branch conceptually — acceptable (the sweep measures legs
independently of serving), noted here so a reader does not mistake the sweep
for a serving contract.
