> Change class: **L** cross-cutting (multi-module, migration, security boundary, public API). See `proposal.md`
> for *why* and *what*; this covers *how*.

## Context

Two facts established against the live database govern everything below, and they supersede every earlier guess.

**The deployed database is Timescale Cloud, PostgreSQL 18.0.4, reached over TLS.** The `timescale` service in
`docker-compose.yml` has never been up in this working copy, so "check what the image ships" answers the wrong
question. On that instance `vector` 0.8.2 and `vectorscale` 0.9.0 are installed; `pg_textsearch` 1.3.0,
`pg_trgm`, `unaccent`, `uuid-ossp` and `vchord` are available but **not installed**; `vchord_bm25` and
`pg_search` are not available at all. Access methods present are `diskann`, `hnsw` and `ivfflat` — there is no
`bm25` access method yet, because the extension that would register it is not installed.

**The database was stamped, not migrated.** `alembic_version` holds one row, `0004`, so the chain claims
`c0c17c6eb1cc → 2bc7726317f6 → 8a7d9b1c2e3f → 9f4a1b7c6d2e → 0001 → 0002 → 0003 → 0004` are all applied. The
live inventory is 16 tables and every one of them is billing or audit. Cross-referencing the inventory against
the `create_table` calls in each revision marked applied:

| Revision marked applied | Claims to create | Actually present |
|---|---|---|
| `c0c17c6eb1cc` | `chat_messages`, `chat_sessions`, `document_vectors` | none |
| `2bc7726317f6` | (renames a column on `document_vectors`) | target absent, never ran |
| `8a7d9b1c2e3f` | `search_documents`, `search_chunks` | none |
| `9f4a1b7c6d2e` | `parent_documents`; ALTERs `clauses` | none; `clauses` absent |
| `0001` | `outbox_events`, `dead_letter_events` | **none** |
| `0002`–`0004` | the 15 billing relations | all present |

Only `0002`–`0004` genuinely ran. `a71f0d7d9c12` — the revision that creates the unified `documents` and
`chunks` relations — is the **other head and is not stamped**, so unlike the rest of that branch it will
actually execute on the next upgrade.

**There is no data to migrate.** Every relation this change creates holds zero rows because it does not exist.
Nothing in this change backfills, copies or drops data, and no `DROP TABLE` appears anywhere in it.

**The most severe consequence is on public surface, and it is not the document schema.** `outbox_events` is
created only by the stamped-but-unapplied `0001`. `POST /auth/forgot-password` and
`POST /auth/resend-verification` are mounted, public and rate-limited; both persist a reset or verification
token and *then* write an outbox event with no exception handling, so both return `500` today after a partial
write that no email will ever complete. The document upload path writes the same relation, but its failure is
currently masked by the identity defect firing first. The relay that reads the relation, by contrast, fails
**soft**: a catch-all in its startup scan swallows the missing relation and the application boots, leaving the
outbox silently and permanently dead behind two warning lines. That resilience is unintentional — it survives
only because of a broad `except` that any exception-tightening pass would remove — and it is the clearest
example in the repository of the invisible-failure pattern this change exists to reduce.

Alongside the schema work sit three defects on already-mounted surface (identity resolution read from unset
request state; profile handlers reading application state under names startup never assigns; database consumers
reading a raw, credential-less configuration value) and roughly 2,900 lines of provably unreachable code.

## Goals / Non-Goals

**Goals:**

- One migration head, and one revision that reads as the complete definition of the target schema.
- Every relation a live read or write path names exists in the deployed database after an upgrade — the outbox
  relations first, because they are the ones failing publicly.
- The already-shipped surface answers correctly: `401` instead of `500` for missing credentials, `503` instead
  of an attribute error for an absent optional client, and a health report that shows a degraded dependency.
- One place to obtain a database URL, serving every driver flavour its consumers need.
- Schema comparison becomes trustworthy, so changes 1 and 2 can use it without it proposing to drop live
  relations.
- Every deletion is one commit with its coupled configuration edit, and no commit leaves the app unbootable.

**Non-Goals** — each of these is a recorded gap with a named owner, not an omission:

- **Memory decay, curation and dedup (D10).** The repository's only decay formula lives in the reconciliation
  module this change deletes, and it is not being replaced. Change 4 carries the durable record; recorded here
  because the capability physically leaves the repository in *this* change.
- **A shared vector-store singleton on application state (dispositions, item 138 residue b — DROP).** No such
  state exists and none is being added; retrieval is raw SQL against the vector and keyword indexes, and a
  store object would create a third retrieval path.
- **`check_cognee`.** Item 198.2 is narrowed: a graph-memory probe already exists — at
  `src/app/middleware/health_check.py:83-90`, *not* at `features/health/` as the disposition ledger records it.
  What is missing is that the versioned endpoint clients call does not report it, which this change fixes. A
  Cognee probe is change 4's step 8 and is out of scope here.
- **Re-enabling the commented graph wiring (D17).** The unwired Saul graph was deliberate and stays commented.
  No requirement here re-enables it, and no flag defaulting to on is introduced. The health probe reports the
  resulting degradation rather than hiding it; `app.state.saul_graph`, `langgraph_checkpointer` and
  `ingestion_graph` remain unwritten, owned by changes 1 and 3.
- **The shadow `shared/agents/**` tree.** Deleting it here is an `ImportError` at boot: a live registry's eager
  imports reach a tool module that imports the 30-byte shadow. Change 3 retargets the importers first.
- **Two phantom imports** — the entity extractor's `graphiti_graph` and the advanced RAG agent's
  `ingestion.embedder` — sit in change 1's blast radius and are not touched here.
- **Narrowing the outbox relay's catch-all handlers.** Sequenced deliberately: narrowing them before the
  relations exist converts a silent degradation into a boot failure, because the wrapper around relay startup
  does not catch database errors. It is also not merely a code change — the `typed-exception-handling`
  capability currently *sanctions* a broad catch at outbox relay degradation boundaries, so narrowing it is a
  requirement change that belongs with the pass that performs it.
- **The second connection pool in the auth service.** It uses the right URL source but builds and disposes its
  own pool per operation. Deferred to change 1 as dependency plumbing, on the record rather than unremarked.
- **The missing binary driver for the checkpointer.** Its driver cannot load libpq at all, so the checkpointer
  short-circuits. Adding it is change 1's step zero; this change only writes the accessor the checkpointer will
  need and corrects the guidance that currently points at the wrong URL flavour.
- **Squashing the history into a single honest baseline.** This is the only fix that makes the chain stop
  misrepresenting itself, and the user rejected it for this change. It becomes possible once the target schema
  has settled; recorded as a post-refactor candidate.
- **Deployment configuration.** The compose file bind-mounts an initialisation script that does not exist, and
  there is no worker or beat service for the task queue while the Makefile invokes a module that does not
  exist. All real, all recorded; a deployment change would break this change's independent committability.
- **The 18 export errors in the MCP core package**, part of the remaining lint baseline, untouched here.
- **Whether the keyword-ranking extension registers an access method literally named `bm25` (F8).** Not
  resolvable read-only — it requires creating the extension, which is DDL against the user's live database.
  Assigned to change 1's step zero against a scratch database. It is *not* change-0 work even though it gates
  this change's deployed-database upgrade; see Risks.
- **Who ran the original stamp (F11).** Unknowable from the repository. Recorded so the next reader does not
  assume the chain was honestly migrated.

## Decisions

### D-1 — Join the two heads with a merge revision, then add exactly one new revision (D14)

The user chose this shape. The merge revision has an empty body: both branches declare the same parent and
touch disjoint relations, and both independently create their extensions idempotently, so there is nothing to
reconcile.

*Alternatives considered and rejected by the user, recorded so they are not revisited:*

- **Editing the unapplied revisions in place** (`8a7d9b1c2e3f`, `9f4a1b7c6d2e`, `a71f0d7d9c12`) — for instance
  prepending the missing `CREATE TABLE clauses` to the revision that ALTERs it. Rejected: although those
  revisions never ran *here*, we cannot prove no other environment applied them. This also removes the
  previously-planned "give `clauses` a create so the chain runs on a clean database" step from this change.
- **A full rebaseline or squash.** Rejected: the 15 billing relations that genuinely exist would have to be
  reconciled by hand against a rewritten root.

*The accepted cost, stated plainly.* The chain permanently misrepresents itself. Revisions recorded as applied
that in fact created nothing stay that way: `c0c17c6eb1cc`, `8a7d9b1c2e3f`, `9f4a1b7c6d2e` and `0001` — D14
names three; the complete live inventory raises it to four, and `2bc7726317f6` is a fifth whose target never
existed. Reversing the chain below the joined head is therefore unsupported: those reversals drop relations
that were never created. The merge revision's docstring names every phantom relation and states the reversal
prohibition, because the docstring is where the next reader will look.

### D-2 — The new revision is authoritative for the target schema, and idempotent (D14, D16)

It defines the whole target shape in one place: the event-outbox relations first, then the unified `documents`
and `chunks` relations with their uniqueness constraints, their generated search column, the vector, keyword and
fuzzy indexes, and the extensions those indexes require. `chunks` gets the `updated_at` column it lacks (D16) —
not cosmetic: it is the only way a later re-embedding campaign can distinguish a current-generation embedding
from a carried-over one, which bears directly on change 1's embedding-dimension work. The trigram index and
every other index change 2 needs are created here; **change 2 ships no DDL, so this change owns all of it.**

The outbox half is ordered first inside the revision and justified independently: it is the half that repairs
two `500`ing public endpoints.

The DDL is written as raw, `IF NOT EXISTS`-style statements rather than the ORM-driven operations, for two
reasons that are both load-bearing. First, `a71f0d7d9c12` is unstamped and will execute on the next upgrade,
creating `documents` and `chunks` before this revision runs; non-idempotent DDL would fail with a duplicate
relation. Second, an inspector-based guard needs a live connection and therefore raises when rendering an
upgrade offline, which would destroy the only proof in this change that needs no database.

*Alternatives considered:*

- **Drop and recreate the relations, for unambiguous authority.** Rejected on the user's own standard: the same
  reasoning that rejected editing unapplied revisions — we cannot prove no other environment applied them and
  holds rows — forbids dropping relations those revisions created.
- **Create only the outbox relations and leave the document schema to a later change.** This was proposed on a
  later planning pass and is overruled: D14 and D16 place the target schema in this change, D15 settles the
  schema contract *before* this migration is written, and change 2 ships no DDL, so there is nothing that would
  later drop what this creates.
- **Create all eleven stamped-but-absent relations, so history and database converge.** Rejected: nine of them
  have no reader, and creating relations nobody reads re-establishes the disease in mirror image — DDL without
  a reader instead of a reader without DDL.
- **Generate the revision by schema comparison.** Rejected: with the models newly registered and the relations
  absent, comparison emits a create for everything, including the relations changes 1 and 2 retire.
  Registration exists to make *future* comparisons safe, not to author this one.

### D-3 — Registration is for comparison safety, not endorsement

Two models — the parent-document and clause models — are declared against a private registry inside the module
this change deletes, so no import can register them on the shared one. They are moved onto the shared registry
before the deletion, and the document and search model modules are added to the migration environment's import
block (the billing modules were already added by concurrent work). Registering them prevents a future comparison
from proposing to drop them; it does not bless them. Change 2 decides the clause and search relations' fate, and
deletes both the model and its registration entry in the same commit.

*Alternative considered:* an allow-list filter in the migration environment instead of moving the models.
Rejected: it requires hand-maintaining a relation-name list forever, while the move is self-maintaining.

Also deleted here: the migration environment's fallback around its metadata assignment, which cannot be reached
because every import sits above it. An unreachable handler that would have reported a broken registration is
worse than none.

### D-4 — Identity comes from token claims, and the two fixes ship together

The repository already has the right seam: a dependency that decodes and validates the access token and returns
its claims with no database round trip. All four unguarded identity readers are rewritten onto it; the fifth is
a guarded branch in an unmounted router, and that branch is deleted rather than kept alive by introducing a
writer for state nothing assigns.

The behaviour change is deliberate: unauthenticated calls to the six mounted document endpoints move from `500`
to `401`. **Ordering, load-bearing:** doing this without creating the outbox relations does not repair the
upload endpoint — it moves the `500` from the dependency layer down to the event insert. The identity fix
therefore lands in or after the commit that creates the relations, never before, and the spec states the
end-to-end outcome rather than the layer-local one.

*Alternative considered:* authentication middleware that populates request state, keeping the existing reads.
Rejected: it puts identity resolution outside the dependency graph, so no endpoint can declare that it needs an
identity, and every unauthenticated path would have to be excluded by URL matching.

### D-5 — Absent optional clients produce 503, not a rename

The profile handlers read application state under two names startup never assigns; startup publishes different
ones. A bare rename converts an attribute error into a `None` that fails later and further from the cause,
because startup sets one of those clients to absent on failure while the annotation promises a value. The reads
are therefore resolved defensively and answer `503` when the client is absent — the shape another feature's
dependency already uses.

*Alternative considered:* rename the startup writes instead of the reads. Rejected: the startup names are read
in several other places, so renaming them is a wider edit with no benefit.

### D-6 — One accessor, three flavours

The existing accessor is correct and is not the defect: it rewrites the scheme, strips the parameters the async
driver rejects as query arguments, and injects the missing credential. The defect is that consumers bypass it —
two read the raw, credential-less configured value, and a third derives a variant by string-editing the
accessor's output, which is then string-edited *again* downstream.

There are three flavours to serve, not two: the asynchronous ORM dialect for the application's pool; a plain
connection URL for consumers that connect directly with a low-level driver, retaining the transport-security
parameter those drivers want; and the value handed to the embedded third-party component. One function
returning one string cannot serve all three, so the accessor gains an explicit flavour selection (or named
accessors per flavour) and every consumer is moved onto it. The guidance in the checkpointer module is corrected
in the same pass: it currently recommends the async dialect, which that consumer's driver cannot accept, while
naming a raw value that carries no credential — wrong in both directions, and change 1 would otherwise follow
it.

Three latent defects in the accessor are fixed while it is open, all one-liners: the credential is interpolated
into the URL without percent-encoding, so a rotation onto a credential containing reserved characters silently
produces a malformed URL; the injection is skipped by comparing the configured credential against a placeholder
literal; and a port is appended to a value that already contains one on the branch taken when the URL has no
username.

*Alternative considered:* repair the raw configuration value once in settings, so even a bypass is safe.
Rejected: the flavours are mutually exclusive — a single stored string cannot be simultaneously correct for
three drivers — so the choice has to happen at the point of use.

### D-7 — Health: degraded is not down

The versioned health endpoint is mounted on two API versions and its checks model forbids unknown fields, so
every addition changes two published shapes at once. The change is additive only, and an absent optional
dependency reports `not_configured` without altering the overall status or HTTP status code — mirroring exactly
how the existing graph-database check is already treated. Without that rule, every environment without the
optional dependency starts returning `503` from a mounted endpoint.

*Alternative considered:* treat an absent dependency as unhealthy, for strictness. Rejected: it makes a
deliberate configuration choice indistinguishable from an outage, and it would fail the probe on every
developer machine.

### D-8 — Deletions carry their coupled edits in the same commit

Nine trees and files are deleted. Four have coupled edits, and two of those couplings are absent from the
deletion manifest: the shared package's `__init__` imports and re-exports the vector-store package, so deleting
the directory alone is an `ImportError` on every module in the application; and a per-file lint-ignore key names
a file inside the reconciliation tree, which leaves no signal at all when it goes stale. Each deletion and its
coupled edit is one commit, so no commit leaves the application unbootable.

Two deletions carry an ordering constraint: the module that declares the parent-document and clause models must
be harvested before the deletion removes it, and the reconciliation deletion must not precede that harvest.

*Alternative considered:* one sweeping deletion commit. Rejected: with zero test coverage on every deleted
tree, a single commit gives one undifferentiated signal, and the import probe that stands in for tests can no
longer attribute a failure.

### D-9 — No spec delta for the deletions or the annotation fix

The deletions and the two `object`-annotation fixes change no externally visible behaviour, so no requirement
is invented to cover them; they appear in the task breakdown and in the Migration Plan only. The disposition
ledger's item 199 is also corrected: the constructor it names was already fixed, and the genuine residue is two
other parameters in the same feature. A third `object` annotation is correct as written — it accepts genuinely
unknown input — and is left alone.

The existence of the outbox relations is already required by the `transactional-outbox` capability, which states
that both relations exist once migrations run. This change makes reality match that requirement rather than
restating it, so no delta is added there.

## Risks / Trade-offs

- [**The unstamped document revision executes on the next upgrade and creates a keyword index using an access
  method that may not exist under that name (F8).** If it fails, its transaction rolls back, head does not
  advance, and the outbox repair — the most severe live break — never runs, held hostage by an unrelated
  index.] → Resolve F8 against a **scratch** database before upgrading the deployed one: create the extension
  there, confirm the access-method name, and render the upgrade. If the name differs, drop the keyword index
  from the authoritative revision and record it as a gap owned by change 1, so the outbox repair is not blocked.
  As a documented fallback, the deployed database can mark that revision applied without running it and let the
  authoritative revision build the whole schema — at the cost of one more revision recorded as applied that
  created nothing.
- [**Creating extensions on a managed instance may require a privilege the application's role lacks.** The
  needed extensions are available but not installed, and a failed `CREATE EXTENSION` fails the revision.] →
  Same scratch-database dry run establishes it, with the same role the deployment uses. This is a precondition
  of the deployed upgrade, not a discovery to make during it.
- [**A plain upgrade from an empty database still cannot reach the target schema**, because the revision that
  ALTERs a relation nothing creates sits in the lineage and the fix for it was rejected. Continuous integration
  therefore cannot build a schema from scratch by upgrading.] → The documented fresh-environment procedure marks
  the phantom branch as applied and upgrades from there, which does reach the target schema; the procedure names
  every revision it skips and what each would have created. The permanent fix is the squash, deferred.
- [**Deleting roughly 2,900 lines with zero test coverage produces zero test signal.** Green afterwards means
  nothing was checked, not that nothing broke.] → Every deletion is proved by an import probe over all six boot
  entry points including the worker package, plus an emptiness search, plus an unchanged test-failure set. If
  any count moves, something imported the deleted tree unexpectedly — that is a finding, not noise.
- [**The identity fix changes a mounted API's failure mode from `500` to `401`.** A client treating the `500` as
  retryable now sees a terminal failure.] → Flagged BREAKING in the proposal. It is the delta, not a side
  effect.
- [**The health response shape changes on two API versions simultaneously, and the checks model forbids unknown
  fields.**] → Additive only, never a rename; the field count and the overall status for the all-absent case are
  both asserted, with the pre-change overall status recorded before the edit and required unchanged after.
- [**Registering the parent-document and clause models could be read as blessing relations changes 1 and 2
  retire.**] → The module docstring states that registration exists for comparison safety and is not an
  endorsement, and change 2's design repeats it in context. Change 2 decides their fate.
- [**Narrowing the relay's exception handling later will expose the boot failure the broad catch currently
  hides**, if the relations somehow still do not exist.] → Ordering is explicit: relations first, narrowing
  afterwards and outside this change. Recorded as a Non-Goal with the reason, not left to be discovered.
- [**Every gate value inherited from earlier planning is stale.** A concurrent split of the billing feature into
  six packages moved files under the plan while it was being written, and mid-write the same commands began
  failing on partially-renamed modules.] → The first task re-captures every baseline to disk and every later
  proof compares against those files, never against a number quoted in a document. The first task also has a
  precondition: the split must be committed and importable before this change begins.
- [**The lint error count can move for reasons unrelated to this change** as files are deleted, making a
  "no worse than" gate soft.] → Compare the count *and* confirm any drop maps to deleted files rather than to a
  suppressed real error.

## Migration Plan

Ordering is dependency-driven; each numbered group is at least one commit and every commit leaves the
application bootable.

1. **Re-capture every baseline to disk** — tests, lint, types, structural scan, migration heads, spec
   validation. Nothing later in this change is provable without it, because three of the six gates are red
   before this change starts and stay red after it. Precondition: the concurrent billing split is committed and
   both the application and its second API version import cleanly.
2. **Delete the unparseable draft** first, because its proof is a *drop* in the lint baseline, which validates
   that the baseline files are trustworthy.
3. **Join the two migration heads.** The merge revision's docstring names every phantom relation and states
   that reversal below this point is unsupported.
4. **Add the authoritative revision** — outbox relations first, then the document schema, the `updated_at`
   column, the indexes and the extensions. Verify by rendering the upgrade offline, then apply it to a scratch
   database, and only then to the deployed one.
5. **Harvest the two models onto the shared registry, register the document and search model modules, and
   delete the unreachable fallback.** Do *not* generate a migration by comparison afterwards.
6. **Delete the remaining dead trees**, each with its coupled edit in the same commit: the inverted parser, the
   vector-store package with the shared package's import and re-export, the empty orchestration package, the two
   empty feature packages with the feature package's import list, and the reconciliation subsystem with the
   worker package's imports and re-exports and the stale per-file lint-ignore key.
7. **Fix the profile handlers' state names** and their absent-client behaviour.
8. **Fix identity resolution repo-wide** — in or after the commit from step 4, never before it.
9. **Move every database consumer onto the accessor**, add the flavours, correct the misleading guidance, and
   fix the three latent accessor defects.
10. **Add the graph-memory dependency to the versioned health report**, after step 7, so the probe reads a
    state surface whose contract is already correct.
11. **Fix the two `object` annotations** and the logging import that raises on every dimension mismatch.
12. **Final gate** — every rung compared against the files from step 1, none by exit code.

Rollback: steps 2, 5–11 are ordinary reverts. Step 3 is revertible by deleting the merge revision *only while
no environment has upgraded past it*. Step 4 is not revertible by reversal — its reversal deliberately does not
drop the outbox relations, and the document relations it creates are owned by another revision's reversal. Roll
back step 4 by restoring from a snapshot, which is safe precisely because there is no data.

## Open Questions

- **Can the `401` be asserted by an automated test in this change?** There is no working test client fixture —
  thirteen collection errors involve it — so the assertion is made by a direct probe rather than a test.
  Repairing the fixture is a test-infrastructure task with its own owner and does not change these specs or the
  task breakdown either way.
- **Should the fresh-environment procedure be promoted from a documented sequence to a committed script or
  build target?** The procedure itself is settled and documented here. Committing it as an executable step is a
  tooling decision that becomes moot if the history squash lands, so it is deliberately not resolved now.
- **Does the deployment role have permission to create the required extensions on the managed instance?** The
  scratch-database dry run answers it before the deployed upgrade. The task is identical either way; only its
  outcome and the fallback differ, which is why this is deferrable rather than blocking.
