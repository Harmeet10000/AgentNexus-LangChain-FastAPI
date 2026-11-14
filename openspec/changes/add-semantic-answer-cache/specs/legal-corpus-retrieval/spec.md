# legal-corpus-retrieval

## MODIFIED Requirements

### Requirement: Answer cache keys are version-aware

The ask answer cache SHALL derive its key from the query, the tenant, the authorized document
scope, the resolved document version of every scoped document, and the canonical pipeline version.
A cached answer SHALL NOT be served after any scoped document's version or the pipeline version has
changed, and the key derivation SHALL match the derivation the replaced inline path defined. The
same serving boundary binds similarity-matched caching: a similarity-matched cached answer SHALL
be served only when the matched entry was stored under the same authorization partition — the same
requesting user, tenant, authorized document scope, resolved document versions, and evidence
policy — as the current request, so a version, scope, or pipeline change excludes the entry from
similarity serving exactly as it excludes the exact-match key.

#### Scenario: A re-ingested document does not serve stale answers

- **WHEN** a scoped document's version changes and the same query is asked again
- **THEN** the derived cache key SHALL differ and a fresh answer SHALL be produced rather than the
  cached one

#### Scenario: A pipeline bump invalidates cached answers

- **WHEN** the canonical pipeline version changes
- **THEN** every previously cached answer key SHALL be unreachable under the new derivation

#### Scenario: Cache-key derivation is pinned by parity

- **WHEN** the parity suite derives answer cache keys through the graph path and the recorded
  inline contract for the same query and scope
- **THEN** the derivations SHALL be identical

#### Scenario: A similarity match does not cross a version or scope boundary

- **WHEN** a similarity lookup finds a stored entry that matches the query meaning but was stored
  under a different authorized document scope, different resolved document versions, or a
  different canonical pipeline version than the current request
- **THEN** the entry SHALL NOT be served and a fresh answer SHALL be produced
