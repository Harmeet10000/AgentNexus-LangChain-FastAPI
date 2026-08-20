> Change class: **L** cross-cutting (multi-module, migration, security boundary, public API). See `proposal.md`
> for *why* and *what*; this covers *how*.

## Context

Everything below is governed by facts measured read-only against the live database, and they supersede every earlier
guess. Where a fact was measured after an earlier draft of this document was written, the correction is recorded rather
than silently applied — see *Corrections applied after initial authoring*.

**The deployed database is Timescale Cloud, PostgreSQL 18.0.4, reached over TLS.** The `timescale` service in
`docker-compose.yml` has never been up in this working copy, so "check what the image ships" answers the wrong
question. On that instance `vector` 0.8.2 and `vectorscale` 0.9.0 are installed; **`pg_textsearch` 1.3.0 is now
installed too** (user-authorized, one statement, scoped — see `findings-database.md` §10), so the access method
`bm25` is present in `pg_am` alongside `diskann`, `hnsw` and `ivfflat`, with operator classes `text_bm25_ops` and
`text_array_bm25_ops`. `pg_trgm`, `unaccent`, `uuid-ossp` and `vchord` are available but not installed;
`vchord_bm25` and `pg_search` are not available at all.

**The keyword-ranking question is closed, and it closed onto a sharper problem.** The repository's existing BM25 SQL
is *already correct* against 1.3.0 — it uses the real two-argument, index-scoped overload of the query constructor,
and its negation and ordering are the expected shape for a distance-style operator. No SQL rewrite is needed. What is
broken is that **no `bm25` index exists anywhere in the database**, and the two-argument overload takes the index name
as a **literal SQL argument**. So keyword retrieval fails until an index exists with exactly the pinned name and the
`bm25` access method, and an index of the right shape under a different name matches nothing while raising nothing.
Index naming is therefore a query contract here, not a convention; see ADR-4, and the requirement
*Retrieval indexes SHALL be created under the exact names the query text names* in `migration-chain-integrity`.


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

**Why the stamp happened, which changes what the repair can be.** `9f4a1b7c6d2e` is not merely unapplied — it is
**unrunnable against any database**. Its `op.create_table` creates `parent_documents`; the remainder of its body
operates throughout on `clauses`: `batch_alter_table` at `:63`, two `UPDATE`s at `:101-102`, three `alter_column`s at
`:103-105`, a foreign key at `:108`, four indexes at `:115-125`, `clauses_bm25_idx` at `:132` and a `diskann`
`clauses_embedding_idx` at `:138`. **No revision creates `clauses` and no ORM model declares it** — `clauses` appears
in exactly one file in the whole versions directory, the revision that mutates it. A real upgrade reaching that
revision dies with `UndefinedTable`. So the stamp was a workaround for a broken migration, not a deployment shortcut,
and the independent proof that the revision never executed is that it is marked applied while `parent_documents` is
absent. This matters because it eliminates one of the two possible repair routes: rewinding the version pointer below
`8a7d9b1c2e3f` and upgrading for real re-enters `9f4a1b7c6d2e` and does not terminate. **ADR-6** records the route
decision — forward repair revision, never a rewind — and why the `clauses` question belongs to the
search-consolidation change rather than to migrations.

**What a real upgrade does today, precisely.** It runs `a71f0d7d9c12` and nothing else: that revision is the only
unapplied one whose `down_revision` the stamped pointer satisfies. So it creates `documents` and `chunks`, and every
other absent relation stays absent, permanently, because `upgrade` skips revisions the pointer already claims. It is
also the revision carrying the extension hazard — it builds a `diskann` index and creates no extension — so the
hazard is not theoretical: it sits on the one revision that actually executes, ahead of both the merge and the repair.

**Only `heads` is a well-defined target while the chain forks.** Measured: `alembic upgrade head --sql` exits **255**
with *Multiple head revisions are present*; `alembic upgrade heads --sql` exits **0** and emits 697 lines. Three
committed call sites use the singular form and are broken today — `Makefile:39`, `README.md:272` and
`.github/workflows/test.yml:105`. The merge revision repairs all three without editing them, which is the argument for
treating single-head as a checked invariant rather than a one-time fix: a future fork silently re-breaks the same three.

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
  relations first, because they are the ones failing publicly. **"Live" is not left to the reader's judgement:** it is
  defined normatively in `specs/migration-chain-integrity/spec.md`, under *Every relation on a live read or write path
  SHALL exist after an upgrade*, as *named by a code path reachable from a route mounted on a published API version,
  through code that is not itself scheduled for deletion or for retargeting by the sequenced changes*. That
  requirement also enumerates the relations the definition resolves to, and the ones it deliberately excludes with the
  change that owns each. Read loosely, this goal would demand creating relations a later change is about to strand.
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
- **Whether the keyword-ranking extension registers an access method literally named `bm25` (F8). CLOSED, not
  deferred.** It does: `pg_textsearch` 1.3.0 is installed on the live instance, `bm25` is present in `pg_am`, the
  opclasses are `text_bm25_ops` and `text_array_bm25_ops`, and the repository's existing BM25 SQL matches the
  installed signature exactly. Nothing about this is assigned to change 1 any more. What replaced it as an open
  problem is narrower and harder: **no `bm25` index exists in the database at all**, and the query names its index by
  literal, so this change must create the retrieval indexes under exact names. Recorded as a requirement, not as a
  Non-Goal.
- **The relay's silent absorption of a missing relation.** When a relation the relay depends on is absent, the relay
  logs two warnings and the application boots healthy, with the outbox permanently dead for that process's lifetime
  and nothing on the readiness surface showing it. This change creates the relations, which removes the *trigger*; it
  does **not** make the relay report the condition, and it adds no requirement demanding that it do so. The reason is
  in ADR-5: an accepted capability explicitly sanctions the broad catch at this boundary, so a requirement demanding
  loud failure cannot be added without the paired delta retiring that sanction and the code satisfying it — all in
  the change that performs the narrowing. Until then the sanctioning requirement wins, and the outbox's health is not
  observable from the readiness surface. This is a recorded debt with the full shape of its repair written down, not
  an omission.
- **Who ran the original stamp.** Unknowable from the repository. Recorded so the next reader does not
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
- **Rewinding the recorded version below `8a7d9b1c2e3f` and upgrading for real.** Rejected on a mechanical ground
  rather than a preference: the rewind re-enters `9f4a1b7c6d2e`, which cannot execute against any database because it
  mutates a `clauses` relation nothing creates. The route does not terminate. `alembic stamp base` fails even earlier —
  `0002` would try to create the fifteen billing tables that genuinely exist. **ADR-6** records the route decision in
  full, including why resolving `clauses` belongs to the search-consolidation change rather than to migrations.

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

### D-3 — The private registry is retired by deletion, not by harvest

**Corrected count.** An earlier draft of this decision said "two models — the parent-document and clause models — are
declared against a private registry". That is wrong, and the undercount was load-bearing: the private
`DeclarativeBase` carries **six** models — entity, relationship, parent-document, clause, event and memory-version.
An implementer following the earlier text would have moved two and silently left four behind, including one named
`Event` whose relation name is a conceptual near-collision with the outbox event this change is repairing.

**Corrected decision.** The module is not harvested. It is **deleted**, whole, as part of the reconciliation
subsystem — of whose 1,129 lines it is 302. Three facts force this and they all point the same way:

1. **It has zero importers repo-wide.** Nothing in `src/` or `tests/` imports it; the only occurrence outside the file
   is an entry in a generated packaging manifest. So none of its six models is a model "live code depends on", and the
   registration requirement's own wording — *re-declared on the shared one **or removed*** — routes them to removal.
2. **Harvesting them would violate D-2 in the very next decision.** Moving a model onto the shared registry makes it
   visible to schema comparison, i.e. schedules DDL for it. D-2 refuses to create `parent_documents` and `clauses`
   precisely because they have no surviving reader — "DDL without a reader instead of a reader without DDL". Applying
   that principle in D-2 and then contradicting it in D-3 is exactly the kind of local inconsistency this change
   exists to reduce.
3. **The relation names that raw SQL elsewhere still mentions are not evidence of a live model.** The knowledge-base
   ingestion nodes and the search repository name `clauses` and `parent_documents` in raw SQL strings; they never
   import these ORM classes. Those SQL literals are owned by the search-consolidation change, which decides those
   relations' fate. Deleting an unimported ORM declaration changes nothing about them.

What this change *does* register is the document and search model modules in the migration environment's import block
(the billing modules were already added by concurrent work), so the shared registry contains the unified document and
chunk relations, the search relations, the event-outbox relations and the billing relations. Registration prevents a
future comparison from proposing to drop those; it does not bless them, and it is not a licence to author this
change's revision by comparison.

*Alternatives considered:*

- **Harvest all six onto the shared registry** (the honest version of the earlier decision). Rejected on ground 2
  above: it schedules DDL for six relations, four of which nothing anywhere references even in SQL.
- **Harvest the two that raw SQL still names, delete the other four.** Rejected: the SQL does not go through these
  models, so registering them buys no comparison safety for the SQL, while still scheduling DDL. It also splits the
  module's deletion across two changes for no gain.
- **An allow-list filter in the migration environment instead of touching the models at all.** Rejected: it requires
  hand-maintaining a relation-name list forever, whereas deletion is self-maintaining.

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

### D-6 — One accessor, two URL flavours, and discrete fields

The existing accessor is correct and is not the defect: it rewrites the scheme, strips the parameters the async
driver rejects as query arguments, and injects the missing credential. The defect is that consumers bypass it —
two read the raw, credential-less configured value, and a third derives a variant by string-editing the
accessor's output, which is then string-edited *again* downstream.

There are **two** URL flavours to serve: the asynchronous ORM dialect for the application's pool, and a plain
connection URL for consumers that connect directly with a low-level driver, retaining the transport-security
parameter those drivers want. One function returning one string cannot serve both, so the accessor gains an explicit
flavour selection (or named accessors per flavour) and every consumer is moved onto it.

**Correction — there is no third flavour, and the error was upstream of this change.** An earlier draft said "there
are three flavours to serve, not two", the third being the form an embedded third-party memory component expects. That
component takes a **discrete-field configuration object** and has no connection-string field at all
(`findings-database.md` §9 is the full retraction of the claim it rested on). Its call site already passes the
discrete fields, including a working credential; the URL that appeared to be handed to it sits in a second,
similarly-named local dictionary in the same function that is only ever returned and never consumed as configuration.
Building a third flavour would have produced surface with no possible caller. The accessor therefore also exposes the
same underlying values as **discrete fields**, which serves that consumer properly and closes a worse defect the
retraction exposed: the call site currently reads host and database name from settings independently of the accessor,
so it can be pointed at a different database than the application with a valid credential and succeed silently.

The guidance in the checkpointer module is corrected in the same pass: it currently recommends the async dialect,
which that consumer's driver cannot accept, while naming a raw value that carries no credential — wrong in both
directions, and change 1 would otherwise follow it.

Three latent defects in the accessor are fixed while it is open, all one-liners: the credential is interpolated
into the URL without percent-encoding, so a rotation onto a credential containing reserved characters silently
produces a malformed URL; the injection is skipped by comparing the configured credential against a placeholder
literal; and a port is appended to a value that already contains one on the branch taken when the URL has no
username.

*Alternatives considered:*

- **Repair the raw configuration value once in settings, so even a bypass is safe.** Rejected: the flavours are
  mutually exclusive — a single stored string cannot be simultaneously correct for both drivers — so the choice has to
  happen at the point of use.
- **Keep a third flavour anyway, as future-proofing.** Rejected: an unused flavour is untested surface documenting a
  false expectation about what its consumer accepts. ADR-3 closes the flavour set at two by decision, so adding a
  third requires naming a consumer that accepts a URL — which is the check that would have caught the original error.


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

**Seven** deletion groups, enumerated exactly — the count is stated here and in `proposal.md` in the same terms,
because a deletion manifest is what an implementer executes and an earlier draft said "nine" while listing eight:

1. the 783-line unparseable draft (`shared/rag/document_processing/todo_temp.py`);
2. the 36-line inverted parser (`utils/toon_parser.py`);
3. the vector-store package's three zero-byte modules;
4. the zero-byte orchestration-type package (five files, `__init__` included);
5. the zero-byte `knowledge_base` feature package (seven modules);
6. the zero-byte `web_scraping` feature package (eight modules);
7. the 1,129-line reconciliation subsystem — its 618-line package, its 209-line worker module, and the 302-line
   private-registry schema module that D-3 now deletes rather than harvests.

Four groups have coupled edits, and two of those couplings were absent from the original manifest: the shared
package's `__init__` imports and re-exports the vector-store package, so deleting the directory alone is an
`ImportError` on every module in the application; and a per-file lint-ignore key names a file inside the
reconciliation tree, which leaves no signal at all when it goes stale. The other two couplings are the feature
package's import list (groups 5 and 6) and the worker package's imports and re-exports (group 7). Each deletion and
its coupled edit is one commit, so no commit leaves the application unbootable.

Ordering, restated after D-3's correction: **nothing is harvested before a deletion any more.** The earlier constraint
— harvest the parent-document and clause models before the reconciliation deletion removes their module — is void,
because D-3 now deletes that module with all six of its models. The one ordering constraint that remains points
*outward*: the shadow agents tree is deliberately **not** in this list, because deleting it before its importers are
retargeted raises `ImportError` at boot.


*Alternative considered:* one sweeping deletion commit. Rejected: with zero test coverage on every deleted
tree, a single commit gives one undifferentiated signal, and the import probe that stands in for tests can no
longer attribute a failure.

### D-9 — No spec delta for the deletions or the annotation fix

The deletions and the two `object`-annotation fixes change no externally visible behaviour, so no requirement
is invented to cover them; they appear in the task breakdown and in the Migration Plan only. The disposition
ledger's item 199 is also corrected: the constructor it names was already fixed, and the genuine residue is two
other parameters in the same feature. A third `object` annotation is correct as written — it accepts genuinely
unknown input — and is left alone.

**A delta *is* added to `transactional-outbox`, and this section previously said the opposite.** The heading's claim
is scoped to the deletions and the annotation fixes only; it never covered the outbox capability, and an earlier
draft's sentence asserting that "no delta is added there" was false while this change shipped one. The correction, in
full:

- The change ships **two MODIFIED requirements** against `transactional-outbox` — its *Outbox Table Schema*
  requirement and its *Migration* requirement — because both are stated in the deployed spec in terms that reality
  contradicts, and a MODIFIED delta is the only mechanism by which a change can restate a deployed requirement.
- It ships **no ADDED requirement** there. An earlier draft added one — *a missing outbox relation SHALL fail
  loudly* — and it is **withdrawn**. Grounds: it had no implementing step in this change, and it directly
  contradicted an accepted requirement in `typed-exception-handling` that sanctions the relay's broad catch. A
  requirement with no implementing step that contradicts a deployed spec is worse than a recorded gap, because it
  archives into `openspec/specs/` as the spec of record and makes the deployed pair mutually unsatisfiable.
- The gap that requirement was reaching for — the relay silently absorbing a missing relation — is recorded as an
  explicit **Non-Goal** above, and the precedence question it raises (*which spec wins until the narrowing lands?*)
  is decided in **ADR-5**: the sanctioning requirement wins, and the narrowing change must ship the relations, the
  code and a paired MODIFIED retiring the sanction together.

The general point, since it is the reusable half: a change's own prose is not evidence about its own deltas. This
sentence was false for as long as it took someone to open the delta directory and count the files.

## Risks / Trade-offs

- [**The unstamped document revision executes on the next upgrade and builds a vector index whose access method no
  revision in its own lineage installs.** `a71f0d7d9c12` builds a `diskann` index; `diskann` is supplied by the
  `vectorscale` extension, which that revision does not create. The revision that *does* create `vectorscale` —
  `8a7d9b1c2e3f:26` — sits on the **other** side of the `2bc7726317f6` branch point, so it is not an ancestor of
  `a71f0d7d9c12`; and on the deployed database it was stamped rather than executed, so its body never ran there
  either. Because `a71f0d7d9c12` is ordered **ahead** of both the merge and the authoritative revision, the
  authoritative revision cannot repair this: a failure there aborts the upgrade before the outbox repair runs, and
  the outbox repair is the most severe live break in the change.] → Measured on the deployed instance,
  `vectorscale` 0.9.0 **is installed** and `diskann` is present in `pg_am`, so the deployed upgrade does not fail on
  this. That makes it a latent chain defect rather than a live blocker, and it is treated as a **precondition** of
  the upgrade rather than as a satisfied dependency: the extension's presence is asserted before the upgrade runs, on
  every instance, and creating it explicitly is the documented remedy where it is absent. The fresh-environment
  procedure states it as an extension precondition of the first revision it does not skip.
- [**F8 as originally written is closed, and the risk it named no longer exists.** The earlier draft of this bullet
  worried that the keyword index's access method might not exist under the name the revision uses.] → `pg_textsearch`
  1.3.0 is installed, the access method is `bm25`, and the existing keyword SQL is already correct against it. What
  replaces this risk is narrower and is recorded as its own requirement: no `bm25` index exists anywhere in the
  database, and because the two-argument keyword-ranking constructor takes the **index name as a literal SQL
  argument**, an index built under any other name matches nothing and reports no error. See ADR-4.
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
- [**The private registry's models could have been read as blessing relations changes 1 and 2 retire**, had they been
  harvested onto the shared registry. That risk is retired by D-3: the module is deleted with all six of its models,
  so nothing is registered and nothing is blessed.] → What remains in its place is the mirror hazard, and it
  is why the deletion is the safer branch: moving an unreferenced model onto the shared registry makes it visible to
  schema comparison and so schedules creation of a relation nothing reads — a reader-less relation, which is the
  defect this change exists to close, inverted. Change 2 still decides the fate of the relations those models
  described; it simply no longer inherits a registration to explain.
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

   **The test baseline must be captured as counts, never as an exit code.** `pyproject.toml` puts
   `--cov-fail-under=80` in the test runner's default arguments and coverage stands at 22.16%, so the runner exits
   non-zero even when every collected test passes. Any proof of the form "the suite is green" is therefore
   unexecutable in this repository. Baselines are captured with coverage disabled and compared as
   passed/failed/errored counts against the file written in this step.
2. **Delete the unparseable draft** first, because its proof is a *drop* in the lint baseline, which validates
   that the baseline files are trustworthy.
3. **Join the two migration heads.** The merge revision's docstring names every phantom relation, names
   `9f4a1b7c6d2e` as unrunnable and the `clauses` relation it presupposes, and states that reversal below this point
   is unsupported. The join is also what makes `alembic upgrade head` — singular — resolve again, repairing
   `Makefile:39`, `README.md:272` and `.github/workflows/test.yml:105` without editing them.
4. **Add the authoritative revision** — outbox relations first, then the document schema, the `updated_at`
   column, the indexes and the extensions. **The retrieval indexes are created under the exact names the revision
   that declares those relations already uses**, so the two converge instead of producing two differently-named
   indexes of the same shape; index naming is a query contract, not a migration-local choice (ADR-4), and no
   conforming implementation picks its own names. Verify by rendering **that revision alone, as a range whose start
   is its parent** — not by rendering to head, which starts from base and emits the whole chain — then apply it to a
   scratch database, and only then to the deployed one. **Applying anything to the deployed database is a separately
   authorized act and is not assumed by any later step**; everything through the scratch rehearsal is committable
   without it. Before the deployed upgrade, assert that the extension `a71f0d7d9c12` needs is present, because that
   revision executes first and this revision cannot repair it (ADR-6).
5. **Register the document and search model modules and delete the unreachable fallback.** The six models on the
   private registry are *not* harvested onto the shared one: they have no importer and no live path, and registering
   them would schedule creation of relations nothing reads (D-3). The module that declares them is retired by
   deletion in step 6, as part of the reconciliation group it sits inside — which is why this step no longer carries
   a harvest, and why step 6 no longer waits on one. Do *not* generate a migration by comparison afterwards.
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

- **Can the `401` be asserted by an automated test in this change?** **No — and the reason recorded earlier was
  wrong, so it is restated on measured grounds.** The earlier text said "thirteen collection errors involve it".
  Measured: `pytest --collect-only` collects **90 tests with zero collection errors**. The thirteen are real but they
  are **setup** errors, not collection errors — all thirteen are `fixture 'client' not found`, in
  `tests/integration/test_health.py` and `tests/integration/test_api_deprecation.py`. The figure was mislabelled in
  kind, not wrong in magnitude, and the mislabelling mattered: it made the obstacle sound like a broken test tree when
  it is a single missing fixture.
  Re-decided on the corrected facts, the answer is still the direct probe, on three grounds: **(1)** no `client`
  fixture exists anywhere in the suite, so an endpoint-level assertion requires building test infrastructure this
  change does not own; **(2)** the coverage gate makes the runner's exit code unusable as a proof for *any* task here,
  so even a passing new test would have to be read out of a count comparison rather than a green run; **(3)** the
  automated version is recorded as follow-up owned by test infrastructure. Repairing the fixture changes neither these
  specs nor the task breakdown.
- **Should the fresh-environment procedure be promoted from a documented sequence to a committed script or
  build target?** The procedure itself is settled and documented here. Committing it as an executable step is a
  tooling decision that becomes moot if the history squash lands, so it is deliberately not resolved now.
- **~~Does the deployment role have permission to create the required extensions on the managed instance?~~
  CLOSED.** `pg_textsearch` 1.3.0 was installed on the deployed instance under the project's own credentials, which
  answers the privilege question by demonstration for the extension that was in doubt. `vector` and `vectorscale` are
  already installed; `pg_trgm` and `unaccent` are available and uninstalled, and the authoritative revision creates
  them conditionally. The scratch-database dry run is retained as a rehearsal of the whole upgrade, not as the
  outstanding answer to this question.

## Corrections applied after initial authoring

**Five** corrections to material recorded earlier in this change and in `docs/relay/decisions.md`. Each is binding on
the tasks that follow; where a correction contradicts an earlier document, this section wins. Three of them correct
statements this change itself made — recorded rather than quietly overwritten, because a correction that leaves no
trace teaches nothing and invites the same error again.

### D14.3 — The recorded proof for change 2 was unachievable

`decisions.md` D14 and D14.1 recorded the proof of change 2 as an offline render of the whole chain, asserting that
`alembic upgrade head --sql | grep -c 'CREATE TABLE search_'` returns `0`.

**That proof cannot pass, and never could.** Measured, it returns `2`, and it will always return `2`. Offline
`--sql` rendering has no database to read a current version from, so it starts from **base** and emits every
revision in the chain — including `8a7d9b1c2e3f`, whose body creates those relations. D14 forbids editing that
revision, so the count cannot be driven to zero by any permitted edit. The assertion and the constraint were
mutually exclusive from the moment both were written.

**A correction to this section's own earlier text, which was wrong in the other direction.** An earlier draft here
claimed the from-base render "does not complete at all" — that `9f4a1b7c6d2e`'s ALTER of the phantom `clauses`
relation makes the render abort before reaching the revisions of interest. That is false, and it was asserted without
being run. Measured: `alembic upgrade heads --sql` exits **0**, emits **697 lines** with a single `COMMIT;`, and the
`clauses` ALTERs are present in the output. Offline rendering emits DDL as *text* and never executes it, so there is
no relation for it to fail against; a phantom target is not an error in a render.

**The true property, stated once so nothing downstream inherits either error:** an offline render is not incremental
and cannot be made incremental. It renders **from base regardless of live state**, because there is no database from
which to read `alembic_version`. It does not abort. Every proof in this change or its successors that depends on
offline `--sql` output being *incremental* — reflecting what the deployed database still needs — is invalid, and every
proof that depends on it *aborting* is equally invalid. What offline rendering does prove is single-head resolution
and the content of a named range.

**Re-scoped, and binding on change 2:** the assertion applies to the **authoritative revision's own rendering** —
the SQL that revision alone emits, obtained as a **range render whose start is its parent**, which does work offline
and does emit only that range — and not to a from-base render of the entire chain. The property under test is that
the authoritative revision does not itself create the relations an earlier revision already claims. That property now
has a requirement to live in: `specs/migration-chain-integrity/spec.md`, *The authoritative revision's own rendering
SHALL NOT create a relation an earlier revision creates*, whose second scenario also records that the from-base form
of the same assertion is to be rejected as unmeasurable. It is deliberately **not** filed under the neighbouring
*SHALL NOT claim relations an earlier revision already claims* requirement, which is about **reversal** and could not
have carried a proof about what a revision creates. No wording in this `design.md`, in `proposal.md`, or in any spec
file in this change carries the from-base form — it appeared only in `decisions.md`.

**The general lesson, recorded because it is the more valuable half:** a Proof that was never executed is not a
proof. This one survived three successive plan documents unchallenged because it *looked* mechanical — a pipe into
`grep -c`, an integer, a comparison — and mechanical-looking assertions attract less scrutiny than prose ones. Every
proof in a plan should be assumed unexecuted until an execution is recorded alongside it.

### D14.4 — All four extensions are created explicitly, and trigram search is built

The deployment role's permissions were probed rather than assumed. Result: `tsdbadmin` is **not** a superuser, but it
holds `CREATEDB` and `CREATEROLE`, and it **can** create both `pg_trgm` (1.6) and `pg_textsearch` (1.3.0).

Two consequences follow.

**Retrieval ships three RRF branches, not two.** Trigram search is in scope and gets built: the fused ranking
combines a vector branch, a BM25 keyword branch, and a trigram fuzzy branch. The earlier framing that treated the
fuzzy branch as contingent on an unknown permission is resolved in favour of building it. This also closes the Open
Question above asking whether the deployment role can create the required extensions; the answer is yes, and the
scratch-database dry run confirms rather than decides it.

**The authoritative revision SHALL create all four extensions explicitly, and MUST NOT rely on ambient
availability.** The four are `vector`, `vectorscale`, `pg_trgm`, and `pg_textsearch`. This is not defensive
boilerplate — but the reason recorded in the earlier draft was factually wrong and is corrected here, because the
correct version relocates the hazard and changes who can fix it.

The earlier draft said *"no revision in the chain has ever created it"* of `vectorscale`. **That is false.**
`8a7d9b1c2e3f` creates it, at line 26 of its body, alongside `vector`, `pg_textsearch`, `pg_trgm` and `unaccent`. The
real defect is narrower and worse:

- `a71f0d7d9c12` builds a `diskann` index (its body at lines 97, 100 and 103) and creates **no extensions at all** —
  its head does not include a `CREATE EXTENSION` for `vectorscale` or anything else.
- `8a7d9b1c2e3f`, the revision that *does* create `vectorscale`, sits on the **other** branch from `2bc7726317f6`, so
  it is not an ancestor of `a71f0d7d9c12` and cannot supply its dependency. On the deployed database it is recorded as
  applied but was stamped rather than executed, so its `CREATE EXTENSION` never ran there either.
- `a71f0d7d9c12` is ordered **ahead** of the merge and therefore ahead of the authoritative revision. **The
  authoritative revision cannot repair this defect**: if the `diskann` build fails, the upgrade aborts before the
  authoritative revision runs, and the outbox repair — the most severe live break in this change — never happens.

Measured on the deployed instance, `vectorscale` 0.9.0 **is** installed and `diskann` is present in `pg_am`, so the
deployed upgrade does not fail on this today. That is what makes it dangerous: the chain's correctness is currently
being supplied by the hosting image rather than by any revision that will execute. It will not reproduce on a fresh
environment, on self-hosted PostgreSQL, or on a managed instance whose image differs. The remedies are the two the
spec records — the authoritative revision creates every extension **its own** indexes depend on, conditionally and
before the first dependent object; and the extension `a71f0d7d9c12` needs is asserted as a **precondition of the
upgrade**, since no revision the upgrade will run creates it. The fresh-environment procedure states that precondition
against the first revision it does not skip.

### Validation reality — the red line is pre-existing, not a defect in this change

`openspec validate --all` reports failures that predate this change and are not resolvable by it.

Four of them — `cognee-v1-api`, `noqa-documentation`, `pattern-matching-standard`, and `typed-exception-handling` —
fail for a **missing `## Purpose` section**. This matters structurally: a change's deltas carry *requirements*, and
**nothing in the delta mechanism emits a `## Purpose` header**. `## ADDED Requirements` and `## MODIFIED
Requirements` are the only sections a delta contributes, and neither can introduce a capability-level Purpose into
an already-accepted spec. Those four failures are therefore **unreachable by any change**, including this one, and
must remain in the accepted baseline until someone edits the deployed spec files directly.

The fifth, `transactional-outbox`, fails for a different reason — **no requirement body carries SHALL or MUST**. An
earlier draft of this section claimed this change's delta *fixes* it, so that "at archive time the merged spec will
validate". **That claim was too strong, and it is withdrawn.** The deployed capability carries **six** requirements —
*Outbox Table Schema*, *Outbox Helper*, *Relay Process*, *Relay Lifecycle*, *Dead Letter*, *Migration* — and **all
six** bodies are non-normative today. This change's delta modifies **two** of them, *Outbox Table Schema* and
*Migration*, and supplies the normative keyword in both. That leaves **four** requirement bodies still non-normative
after archive, so `transactional-outbox` stays red. The delta reduces the count of offending bodies from six to four;
it does not turn the capability green, and no one should plan around its doing so. Turning it green means restating
four requirements this change has no reason to touch, which is a direct spec-hygiene edit, not this change's work.

**The sixth failure is a change, not a spec.** `change/mintlify-documentation` is an unrelated open change with its
own author; it is named here only so that the count of six is fully accounted for and nobody attributes it to this
change. Baseline, measured: **21 passed, 6 failed (27 items)**, with `change/cleanup-foundation` among the passes.

**During authoring, the counts do not move.** The delta is applied to the deployed spec only at archive; until then
the deployed file is untouched and still fails. So the correct expectation while this change is open is that the
pre-existing failure set is unchanged, and the acceptance criterion is **no new failures beyond the pre-existing
set** — not "validate --all passes". Anyone reading the red line as evidence of a defect in change 0 is reading it
wrong.

### Stale figures inherited from earlier planning, corrected

Three numbers travelled through several planning documents unchecked. They are corrected here because tasks were
being justified by them.

- **"Thirteen collection errors."** There are **zero** collection errors: `pytest --collect-only` collects **90**
  tests and exits clean on collection. The thirteen are **setup** errors — `fixture 'client' not found`, all in
  `tests/integration/test_health.py` and `tests/integration/test_api_deprecation.py`. Right magnitude, wrong kind, and
  the wrong kind implied a broken test tree rather than one missing fixture.
- **The suite's real state**, captured with coverage disabled: **22 failed, 55 passed, 13 errors**. The gate that
  makes the runner's exit code useless is separate and is not a test failure at all —
  `--cov-fail-under=80` against 22.16% actual.
- **The lint count.** Measured on 2026-08-18: `uv run ruff check src/` → **123 errors**. An earlier note recorded
  120; that figure is not reproducible today and is withdrawn rather than reconciled, because the count moves whenever
  a file is added or deleted and no absolute figure is durable. Every gate in this change compares against a file
  captured in step 1, never against a figure quoted in a document — including this one. What *is* durable is the
  attribution rule: a drop must map to a deleted file, and a drop that maps to a suppressed real diagnostic is a
  regression wearing a green tick.
- **The other gates, same date and same caveat:** `ty check src/` → 46 diagnostics; `ast-grep scan src/` → 4
  error-level findings; `openspec validate --all` → 21 passed, 6 failed.

### A spec can pass validation while being false

`outbox-helper-extraction` validates green today, and an earlier draft of this section had its falsehood **exactly
inverted**. The draft said the accepted spec "describes code that does not exist", requiring an engine built from the
connection URL and disposed in a `finally` block, "while the real helper draws a session from the application's
shared factory and owns no engine at all."

**Measured, both halves are live.** `auth/service.py` takes a pooled session when a session factory is injected, and
otherwise falls through to an `else` branch that does `create_async_engine(get_database_url())` and
`await engine.dispose()` in a `finally` — and that branch is reachable from `auth/router.py`, which supplies no
factory. So the engine-per-call behaviour the accepted spec describes is not a description of vanished code; it is a
description of code that runs today on a mounted public path.

The consequence for this change is the one that mattered: a delta asserting that behaviour "no longer exists" would
have archived a **false statement** into `openspec/specs/` as the spec of record, and the false statement would have
been load-bearing — it would have retired an obligation (*connection pools SHALL be owned by the startup sequence*)
that nothing had actually satisfied. That delta was rewritten. It now states the property that *is* true today, states
plainly that the fallback exists and that this change does not remove it, and carries the residue as a named
outstanding defect with the connection-plumbing change as its owner.

The reusable lesson is narrower than "validation cannot check truth", though that is also true: **the direction of a
spec's error is not guessable from the spec.** A stale spec can be stale because reality moved on, or stale because
the spec described an intention reality never reached. Both look identical on the page, and only reading the code
distinguishes them. This one was misread in the more flattering direction — the direction where the work is already
done.

One mechanical consequence surfaced while writing that delta, and it is worth recording: a `## MODIFIED Requirements`
block **replaces the whole requirement**, so strict validation refuses any delta that does not reproduce every
scenario title the accepted spec still carries, verbatim. Correcting a stale spec therefore cannot rename its
scenarios — the delta must carry the old titles forward and correct only their bodies. The `outbox-helper-extraction`
delta in this change does exactly that.

**The same mechanic has a sharper edge that strict validation does not catch at all.** Because a MODIFIED block
replaces the requirement wholesale, a block that reproduces *fewer* scenarios than the accepted requirement carries
**deletes the missing ones on archive** — silently, with no `## REMOVED` block, no Reason and no Migration. Strict
validation accepts it, because the block it sees is internally well-formed and the evidence that something is missing
lives in a file the change does not contain. This change's `typed-exception-handling` delta had exactly that defect
through two drafts: it reproduced five of the accepted requirement's six scenarios, dropping
`Reconciliation fetch failure catches PostgresError`.

It is now fixed by **reproducing all six verbatim, in the accepted order**, and the stale one is kept on purpose. The
first attempt at a fix disclosed the omission in prose inside the requirement body — which was honest but still
deleted the scenario on archive, because prose is not a mechanism. `## REMOVED Requirements` is not the alternative
either: it operates at requirement granularity, so naming this requirement there would retire the whole asyncpg
guarantee including the five scenarios that must survive. There is no scenario-level REMOVED. Retiring one stale
scenario is therefore a direct edit to the accepted spec, and it is routed to the same spec-hygiene pass the four
`## Purpose` failures need.

Two rules fall out of this, both cheap and both learned the expensive way. **Count the accepted requirement's
scenarios before authoring a MODIFIED block, and reproduce every one.** And **do not treat a green
`validate --strict` as evidence that a delta is complete** — it is evidence that the delta is well-formed, which is a
much weaker claim, and the gap between the two is exactly where a deployed guarantee can disappear without an author.

## Architecture Decision Records

Four ADRs are cited throughout this document. They are recorded here as first-class headings because an ADR outlives
the change that produced it — a decision cited as authority but never written down reads as authority that does not
exist. Each consolidates reasoning stated above; where an ADR and a `D-n` decision overlap, the ADR is the durable
form and the `D-n` is this change's implementation of it.

Numbering starts at 3 deliberately: ADR-1 and ADR-2 belong to earlier changes and are not restated here.

### ADR-3 — The database-URL flavour set is closed at two, and a third requires naming a consumer

**Decision.** Exactly **two** URL flavours exist — the async-ORM dialect and the plain low-level-driver form. The
accessor additionally exposes the same underlying values as **discrete fields**. Adding a third flavour requires
**naming a consumer that accepts a URL**.

**Context.** An earlier draft asserted three flavours, the third being what an embedded third-party memory component
expects. That component takes a discrete-field configuration object and has **no connection-string field at all**;
the URL that appeared to be handed to it sits in a second, similarly-named local dictionary that is only ever
returned and never consumed as configuration.

**Consequence.** Building the third flavour would have produced surface with **no possible caller** — untested code
documenting a false expectation about what its consumer accepts. The naming requirement is the cheap check that
would have caught the original error, which is why it is the operative clause rather than the count.

**Status.** Accepted. Implemented by **D-6**. The retraction of the claim it replaces is `findings-database.md` §9.

### ADR-4 — Retrieval index names are a query contract, not a migration-local choice

**Decision.** Every retrieval index is created under the **exact name** the declaring revision already uses. No
conforming implementation picks its own names.

**Context.** The two-argument keyword-ranking constructor takes the **index name as a literal SQL argument**, pinned
at `src/app/features/search/constants.py:15`.

**Consequence.** An index of the **right shape under a different name matches nothing and reports no error** — the
failure is silent and returns empty results rather than raising. This inverts the usual intuition that index naming
is cosmetic: here it is part of the query's semantics. It also means two differently-named indexes of the same shape
is a real and reachable outcome, which is why the revision converges on the declaring revision's names instead of
choosing fresh ones.

**Status.** Accepted. Implemented in Migration Plan step 4 and asserted by that step's Proof, which compares created
index names against the pinned constant.

### ADR-5 — Until the narrowing lands, the sanctioning requirement wins

**Decision.** Where `transactional-outbox` and `typed-exception-handling` conflict over the event relay's broad
exception catch, **the accepted requirement that sanctions the broad catch wins**. The change that narrows it must
ship the relations, the code, and a paired `## MODIFIED` retiring the sanction **together**.

**Context.** An earlier draft added a requirement to `transactional-outbox` demanding that a missing outbox relation
fail loudly. It was **withdrawn**: it had no implementing step in this change and directly contradicted an accepted
requirement that sanctions the relay's broad catch.

**Consequence.** A requirement with no implementing step that contradicts a deployed spec is **worse than a recorded
gap**, because it archives into `openspec/specs/` as the spec of record and makes the deployed pair **mutually
unsatisfiable** — a state no later change can resolve without knowing which was intended. Hence the precedence rule
and the all-together constraint on the narrowing change.

**Status.** Accepted. The gap it leaves open is a recorded Non-Goal, not silence.

### ADR-6 — The migration repair is a forward idempotent revision; rewind is rejected on mechanical grounds

**Decision.** Repair the chain by **joining the two heads and adding one forward, idempotent revision**. Do **not**
rewind and re-upgrade. The `clauses` question belongs to the **search-consolidation change**, which retargets its
readers rather than creating the relation.

**Context.** `9f4a1b7c6d2e` **cannot execute against any database**: it mutates a `clauses` relation that no revision
creates and no model declares, while its own `op.create_table` creates `parent_documents`. Proof it never ran here —
it is recorded as applied, yet `parent_documents` is absent. **This is why the original stamp happened.**

**Consequence, and it is the reason this is a mechanical rejection rather than a preference.** The rewind route
**does not terminate**: rewinding below `8a7d9b1c2e3f` re-enters `9f4a1b7c6d2e`, which raises. `alembic stamp base`
fails even earlier — `0002` would try to create the fifteen billing relations that genuinely exist. So there is no
sequence of standard alembic operations that reaches the target schema by replay, independent of anyone's preference
about tidiness.

**Accepted cost, stated plainly.** The chain **permanently misrepresents itself**. Revisions recorded as applied that
created nothing stay that way — `c0c17c6eb1cc`, `8a7d9b1c2e3f`, `9f4a1b7c6d2e`, `0001`, and `2bc7726317f6` as a fifth
whose target never existed. Reversal below the joined head is therefore **unsupported**, because those reversals drop
relations that were never created. The merge revision's docstring carries this warning, because the docstring is
where the next reader will look.

**One consequence the forward route does not escape.** `a71f0d7d9c12` is unstamped and executes **ahead** of both the
merge and the authoritative revision, and it builds a `diskann` index while creating no extensions. The authoritative
revision therefore **cannot repair it** — a failure there aborts the upgrade before the outbox repair runs. This is
handled as a **precondition asserted before the upgrade**, not as a dependency assumed satisfied, because on the
deployed instance the dependency is currently supplied by the **hosting image** rather than by any revision that will
execute, and that will not reproduce on a fresh environment.

**Status.** Accepted. Implemented by **D-1** and **D-2**; the permanent fix (a history squash) is deferred.
