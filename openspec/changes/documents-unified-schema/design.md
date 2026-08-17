> Change class: **L**. The proposal covers *why* and *what*; this covers *how*. It does not restate the proposal.

## Context

**The document and retrieval schema is greenfield, not duplicated.** A live probe of the actual database — a
managed PostgreSQL 18.0.4 instance, not the local compose service — found `alembic_version` holding exactly one
row on the billing lineage, sixteen tables, all of them billing or audit. The unified document store, the unified
chunk store, the superseded search document and chunk tables, the clause table, the parent-document table and
the original vector table are **all absent**. The branch through those revisions is stamped as applied while none
of its tables exist: someone stamped rather than upgraded.

Three consequences shape every decision below.

1. **There are not five parallel document schemas in production. There are zero.** So this change cannot be a
   data migration, a backfill, or a drop. Every question that used to read "what do we put in this column for
   existing rows?" has no referent.
2. **The target has never run either.** The mounted document router would fail on an undefined table for every
   retrieval call today; that failure is currently masked because owner resolution raises first. Two live breaks
   are stacked, and fixing only the outer one exposes the inner one. Both belong to change 0. This change must be
   neither credited with nor blamed for either.
3. **All schema authorship moves upstream.** The chosen repair is to merge the two migration heads and add one
   new migration that creates the target schema outright. This change therefore ships **no DDL of its own**. It
   contributes the *column and index specification*; change 0 executes it. That is the same author-first,
   code-after inversion applied to the schema contract that the accepted ADR applies to change 1.

**The write path being removed is unrunnable, not merely unreached.** Its ingest call publishes an outbox event
inside the caller's transaction, and the outbox tables do not exist either — created only by another
stamped-but-never-applied revision. The asynchronous half of that ingest fires only from the event the write
would have emitted. So the counterfactual "someone mounts the router" still ends in a failure on a nonexistent
relation, and the consumer could never have been reached. Creating the outbox tables is change 0's, not ours.

**Two environment facts, both newly verified, both favourable.** The database's application role is not a
superuser but does hold `CREATEDB` and `CREATEROLE`, and it **can** create both the keyword-search extension
(available at 1.3.0) and the fuzzy-match extension (available at 1.6). Both were created inside a transaction
and rolled back during verification; the database was left clean. This closes the last open unknown that could
have changed this change's content: the fuzzy retrieval branch is buildable, so retrieval ships three branches,
not two.

**What is *not* established: none of the three retrieval indexes has ever existed.** The keyword, vector and
fuzzy indexes on the chunk store are *defined* by an unapplied revision, and the fuzzy extension is not
installed. No reader of this design may conclude that keyword, vector or fuzzy retrieval works today, on either
schema. The refutation of the earlier claim that the fuzzy branch "has no target equivalent" is about **schema
definition only**, never about working capability.

## Goals / Non-Goals

**Goals:**

- Name one document store and one chunk store as the sole retrieval truth, and make every retrieval reader agree.
- Invert the import direction so the unified feature no longer imports its own retrieval helpers out of the
  superseded feature.
- Delete the superseded schema-bound twin — models, repository, router, dependency layer, ingest service path,
  ingest DTOs, Celery ingest task — without losing a single read capability and without making anything newly
  reachable.
- Retarget the retrieval graph's fused search off the clause table, which no migration creates.
- Hand change 0 an exact, unambiguous column-and-index specification for the one migration that will create the
  target schema for the first time.
- Leave behind a gate that fails when query text names a database identifier no migration creates — the defect
  class that produced the clause-index hole.

**Non-Goals:**

- **All DDL and migration authorship.** No revision is written here. Change 0 owns the head merge and the single
  authoritative create-schema migration.
- **Dropping anything.** Nothing is dropped, because nothing exists to drop. Any objection of the form "change 0
  creates tables that change 2 drops" is factually void.
- **Creating the outbox tables.** Change 0's migration, and its catch-all exception tightening in the relay, are
  both change 0's.
- **Mounting the superseded search router.** Explicitly out of scope, and gated on the owner-resolution defect.
  The constraint is honoured by *deletion*, which is strictly safer than leaving an unauthenticated router in the
  tree.
- **Replacing the deleted raw-text ingest.** This is a real capability loss *to the codebase* and is recorded
  rather than hidden. The deleted endpoint accepted a title and a body of text from an unauthenticated request
  and never stored a source object, so it could not satisfy the ownership and provenance requirement this change
  adds. Anyone who wants it back must build it as an authenticated document text-ingest endpoint that writes a
  real owner and stores the raw text as an immutable object first so a provenance reference exists. **Gated on
  the owner-resolution fix.** Building it here would create exactly the unauthenticated reachable surface the
  scope constraint forbids.
- **Fixing owner resolution on the mounted router.** Change 0's, and it must be fixed together with the schema
  or an authorization error simply becomes an undefined-table error and reads as progress.
- **Cross-tenant deduplication.** The escape hatch is recorded and deliberately not built: one immutable content
  blob and chunk set keyed by digest, plus a per-tenant pointer row carrying that tenant's metadata and access
  control. The object reference already gives content-addressed storage keys, so the groundwork exists.
- **Retrieval-quality tuning.** The fuzzy-similarity floor is a *first-time calibration*, not a re-tune, and
  belongs to change 1 along with re-ranking, which is the one genuinely missing piece of hybrid retrieval.
- **Plumbing a clause-type filter through the retrieval graph.** The graph's plan object forbids extra fields, so
  the retargeted call passes no clause filter. The hand-rolled question-answering path *does* filter by clause
  type from its request payload, so **the graph path is strictly weaker until change 1**. Recorded rather than
  quietly matched downward.
- **Choosing between the graph-backed retrieval path and the hand-rolled one.** Change 1's, and it needs
  re-ranking settled first. This change's obligation is discharged by pointing the graph at the unified schema
  and leaving both alive.
- **The embedding decisions.** Which normalization convention stored vectors use, and unhardcoding the vector
  dimension, are change 1's. Both are cheapest at change 1's moment because the chunk store has zero rows — one
  convention is simply chosen before the first row is written.
- **The clause-reading tool stub** that returns an empty list: change 3's.
- Recorded gaps adjacent to this change, carried from the disposition ledger so they are not silently dropped: a
  shared vector-store singleton object is **dropped** (it would create a *third* retrieval path alongside the two
  this change unifies); the "refactor vector-store code" item is **merged** into the reader-less derived column
  work, whose other half is a set of zero-byte packages in change 0's deletion manifest; and evaluating external
  retrieval frameworks is **deferred**, since the scope decision commits to the existing extension-based path.

## Decisions

### 1. One new capability, not two, and it covers schema *and* retrieval together

The twenty existing capabilities were enumerated first. None covers documents, chunks, retrieval, schema or
migrations, so a new capability is required.

*Alternatives considered:* **split schema from retrieval** — rejected, it would put the index requirements in a
different spec from the columns they index, and the two cannot be satisfied independently. **Extend an existing
capability** — no candidate fits; the closest by name governs model injection, not storage.

### 2. Reuse the existing model-injection capability rather than leave it stale

That capability carries a requirement bound to the superseded search service and a dependency-layer requirement
naming its deleted dependency module. This change dissolves both subjects. Leaving them would leave the project's
own spec asserting behaviour that has no subject — the spec-level version of the defect this whole change exists
to remove. So one requirement is **REMOVED** with reason and migration, and one is **MODIFIED** to keep only its
surviving half.

*Alternatives considered:* **leave the capability untouched** — rejected, it makes the spec lie. **Rewrite the
whole capability** — rejected as out of scope; its two document-side requirements are unaffected and are the
migration target for the removed one.

### 3. Ship no DDL at all

The superseded tables were never created anywhere, so a drop revision's body would be a no-op in every
environment that exists, while permanently adding a lineage node and a downgrade that recreates tables nobody
wants — an invitation for a future reader to conclude those tables once mattered.

*Alternatives considered:* **a `DROP TABLE IF EXISTS` revision** — rejected for the reason above. **Strip the
superseded DDL out of the unapplied revision that declares it** — rejected by decision, because we cannot prove
no other environment applied that revision; editing it is off the table even though it is stamped-but-unrun here.
**Accepted residue:** a rendering of the full migration history from base therefore still emits the superseded
`CREATE TABLE` statements. See the Risks section, where that is measured rather than assumed.

### 4. Delete the writer; do not invent a tenant

The chunk owner and the document object reference are non-nullable with no default, and the deleted ingest path
supplied neither. The question dissolves rather than being answered: the code being retargeted is the *reader*
side, which already has both values, and the writer side is deleted rather than ported. The surviving upload path
already supplies an owner from the authenticated request and an object reference from the stored object key.

*Alternatives considered, all rejected:*

| Option | Why rejected |
|---|---|
| A sentinel or system owner | Creates a tenant whose rows every per-tenant identity check ignores and whose chunks no owner-scoped lookup will ever return for a real user. A row nobody can retrieve is not data; it is a leak waiting for the first query that forgets to filter. |
| Make the chunk owner nullable | Destroys the tenant-isolation invariant on the only store that has one. Every owner-scoped predicate silently becomes a partial scan over unowned rows. |
| An empty or nullable object reference | The reference is the provenance link back to the immutable stored object. An empty one means the text came from nowhere — unauditable in a legal product, and it removes the only way to re-parse a document after a chunker change. |

Because these columns are being written for the first time, the enforcement point matters more than it would in a
migration: **required, non-optional fields on the ingest contract so the failure is a client error at the edge**,
never a driver-level integrity error, and **never a database default added "for safety"** — that would
permanently weaken the contract with nothing to justify it.

### 5. Document identity is per-tenant, and cross-tenant duplicate storage is accepted

Identity becomes the pair of owner and content digest rather than the digest alone. The same document uploaded by
two tenants is stored twice — two documents, two chunk sets, two embedding sets. **This is the correct trade and
is accepted, not mitigated.** Global deduplication over a multi-tenant corpus leaks in three distinct ways: the
second uploader's ingest returns the first uploader's document identifier, disclosing that someone else holds
that document; the shared row's metadata, jurisdiction and parties come from whoever uploaded first; and an
erasure request from one tenant destroys the other's document. The superseded store had no owner column at all,
so it could not have done better.

*Alternatives considered:* **global digest deduplication** — rejected for the three leaks above. **Content-addressed
storage with per-tenant pointer rows** — the right long-term answer, recorded as a Non-Goal rather than built,
because it changes the read path for every consumer and this change is subtraction.

*Cost, stated plainly:* storage and embedding spend scale with tenants times shared documents. For a corpus whose
shared part is public statutes and standard-form contracts, that is real.

### 6. Chunks record their modification time, and the ingest path writes it

The chunk store gains a modification timestamp its parent document store already has. It has a live consumer, not
a hypothetical one: the ingest path rewrites every chunk row **twice per ingest**, once on first write and again
after verification mutates the rows in place. Without it there is no way to distinguish a chunk whose embedding
was written by the current model generation from one carried over from a prior one — precisely the audit a
re-embedding campaign needs, and embedding-dimension drift is already a known live hazard.

*Alternatives considered:* **drop the tracking** — defensible if chunk rows were immutable-and-replaced, but the
double write proves they are not, so dropping it means recording a known-unauditable mutation. **Rely on the
ORM's update hook alone** — rejected, and this is the trap that makes the decision non-trivial: that hook fires
for update statements but **not** for a conflict-resolving insert. The value must be written into the row payload
*and* into the conflict-resolution set. Miss either and the column exists, is non-nullable, and never changes —
the worst outcome, because it looks maintained and is not.

### 7. Retrieval keeps three branches, including the fuzzy one

The claim that the fuzzy branch has no equivalent on the target is refuted at the level of schema definition: the
target's index and the query that uses it both exist in source. The permission probe then settled the real
question — the application role **can** install the fuzzy extension — so the branch is buildable and stays.
Fuzzy matching is the branch that survives OCR noise and typographical error in scanned legal documents, which is
this product's actual corpus.

*Alternatives considered:* **two branches** — cheaper and was the correct answer had the extension been refused;
kept on the record as the fallback. **Add a second fuzzy index over the raw content column to restore the
superseded behaviour exactly** — rejected: a whole extra index, and write amplification on the hottest store in
the schema, to serve one of three branches in a variant nobody has ever tuned.

*Semantic delta to record:* the superseded branch matched raw content; the target matches a derived text that
concatenates classification, preamble and content. Character-similarity is a normalized ratio over the whole
string, so a long preamble **dilutes** the score of a match located in the content. The similarity floor is
therefore effectively *stricter* on the target for any chunk with a preamble, and because rank fusion uses only
rank position, the visible effect is branch **recall**, not score.

### 8. Retarget the clause readers rather than leave them stale

The retrieval graph's fused-search call targets a clause table that no migration creates, using an index created
on a table created nowhere. Three independent findings, now joined by a fourth — the table does not exist in the
live database and never has — all point at the same hole. Retargeting costs nothing in data or deployment terms.

*Alternatives considered:* **leave it stale** — rejected; it means preserving code that reads a table nothing has
ever created in any environment, which is how the invisible-failure register got this long. **Backfill clause rows
into the chunk store** — rejected; there are no rows, so a backfill would migrate nothing while roughly doubling
this change.

### 9. Move the graph-backed ask path; do not delete it

The deleted service holds the **only caller** of the retrieval graph builder, and that graph is change 1's
foundation. A careless deletion orphans the entire retrieval graph.

*Alternatives considered:* **delete it with the rest of the service** — rejected for the reason above. **Promote it
over the hand-rolled path now** — rejected as change 1's decision; this change moves it, leaves it unexposed by
any router, and prejudges nothing.

### 10. Guard hardcoded identifiers with a static gate, not string interpolation

Both the superseded constraint name and the keyword index name are embedded as literals in query text, and the
target side is no cleaner. The keyword-search extension needs the index named *inside* the query because it reads
that index's corpus statistics, so a rename is a silent runtime break with no lint, type or migration warning.

*Alternatives considered:* **interpolate a constant into the query string** — rejected; it trips the project's
hardcoded-SQL lint rules and buys nothing, because the literal still has to exist somewhere. **Do nothing** —
rejected; this is the exact defect class that produced the clause-index hole. The gate is **red on three counts
today**, which makes it simultaneously the proof of the clause retarget and a permanent guard.

### 11. Relocate helpers behind a temporary re-export shim

The global test conftest imports twenty symbols from the module being deleted, at module level, so a missing
symbol is a collection error for the **entire** suite. The relocation therefore lands with a re-export shim in
place, so both import paths resolve until the deletion and the conftest rewrite land in the same commit.

*Alternatives considered:* **move and delete in one step** — rejected; it breaks collection for every test in the
repository, which is the single highest-risk moment in this change.

## Risks / Trade-offs

- **[The global test conftest is a single point of failure for the whole suite]** — it imports twenty symbols from
  the module being deleted, at module level, and conftest is global, so one missing symbol is a collection error
  for **every** test, not just this feature's. → Relocate behind a re-export shim so both import paths resolve;
  delete the module and rewrite the conftest **in the same commit**; compare collection output against a captured
  baseline. Never delete a module and its conftest reference in separate commits.

- **[This change's SQL ships unexecuted]** — no test in the repository touches a database. The integration test
  for this feature mocks the repository, the embedding client *and* the session, and the conftest has no engine,
  no table creation and no container. So every column, index and constraint name here passes green whether it is
  right or wrong. → The static identifier gate catches name drift for free; the real-database gate is the only
  thing that can prove the stores can be created and queried at all, since they have never existed anywhere. **If
  the real-database gate is descoped, this line is the record that this change's SQL shipped unexecuted.** This is
  the risk most likely to be waved through, and it is the one that produced the clause-index hole.

- **[A full-history rendering still emits the superseded table DDL]** — measured, not assumed: rendering the
  migration history offline today emits **two** `CREATE TABLE` statements for the superseded search tables,
  because the revision that declares them stays in the lineage and editing it is rejected. → Two guards, and an
  honest limit. The guards: the source tree names no superseded table outside migration history, and the
  authoritative create-schema migration creates neither. The limit: the count over the *full* history from base
  cannot reach zero without either the rejected edit or a no-op drop revision, so **it does not reach zero and
  this change does not claim it does.** The exposure is narrower than it looks — a from-base provisioning already
  cannot complete for an unrelated reason, a phantom alter against the clause table — and closing that is change
  0's, together with making its new migration authoritative rather than additive.

- **[The chunk modification timestamp exists but is never written]** — the ORM's update hook does not fire for a
  conflict-resolving insert, which is the only way chunks are written. → The value must appear in **both** the row
  payload the ingest path builds **and** the conflict-resolution set; the verification greps for both. A
  non-nullable column that never changes is worse than no column.

- **[No retrieval index has ever been created, on either schema]** — all three branches' indexes are declared by
  an unapplied revision, and the fuzzy extension is not installed. → The permission probe proves they are
  buildable; the real-database gate is the only thing that can prove they build. Recorded here so no later reader
  mistakes "the target defines an equivalent" for "the branch works today."

- **[Change 1 lands first and builds a promoted pipeline writing the abolished schema]** — its persistence nodes
  target clause, parent-document, entity and relationship tables while this change names the chunk store the sole
  retrieval truth. → The accepted schema ADR is authored **before** change 1 implements those nodes, so change 1
  writes the chunk store from the start and never touches the clause table. The ADR is a gate on another change,
  not a footnote.

- **[Two changes edit the same task registry and migration environment files]** — change 0 rewrites both for its
  own deletions and model registration; this change edits adjacent lines in each. → Change 0 lands first and this
  change edits the residue. Explicit coordination point: change 0 must register the unified document models and
  must **not** register the deleted search models, or every migration command dies with an import error.

- **[Lint, type and test counts move for reasons unrelated to correctness]** — deleting roughly eleven hundred
  lines changes every count, and a source file that does not parse means the tree is not fully analysable today.
  → Every verification states `<=` against a baseline captured on a clean tree, never an absolute number. And
  never the process exit code: the coverage floor in the test configuration makes a fully green suite exit
  non-zero, so only the printed summary line is meaningful.

- **[The retargeted graph path is weaker than the hand-rolled one]** — the graph's plan object forbids extra
  fields, so the retargeted fused search passes no clause-type filter, while the hand-rolled path filters by one
  from its request payload. → Recorded rather than quietly matched downward; plumbing a real clause filter through
  shared graph state is change 1's.

- **[The superseded feature has been restructured before without obvious history]** — its compiled bytecode
  directory holds entries for source files that no longer exist, evidence of an earlier rename nobody reconstructed.
  → Noted because it slightly weakens any "nobody ever used this" argument. The argument this change actually
  relies on does not need it: the write path is unrunnable on four independent grounds, the strongest being that
  both the target table and the dispatch table it writes through do not exist.

- **[Deleting the ingest path removes the repository's only raw-text ingest]** — a genuine capability loss to the
  codebase, though not to users, since the endpoint was never mounted. → Recorded as a Non-Goal with the exact
  shape of its replacement and the gate it waits on. This change must not build it: doing so would create exactly
  the unauthenticated reachable surface the scope constraint forbids.

- **[Per-tenant identity makes duplicate storage the normal case]** — storage and embedding spend scale with
  tenants times shared documents. → Accepted deliberately over a cross-tenant information leak; the
  content-addressed escape hatch is named as a Non-Goal so the trade is revisitable rather than forgotten.

## Migration Plan

**No database migration is authored here.** What follows is the ordering of source changes, the specification
handed upstream, and how each is proven.

### Baselines to capture before the first edit

Capture, on a clean tree, the test summary line, the collection summary, the lint count, the type-diagnostic count
and the formatter check. Every later verification compares `<=` against these files rather than against an
absolute number, and against the printed summary rather than the exit code.

### Schema inputs handed to change 0

Change 0's single authoritative create-schema migration must create exactly the following, and nothing else in
this area. Identifier names are load-bearing: the keyword index name and the chunk uniqueness constraint name are
embedded in query text and asserted by this change's static gate, so a rename is a silent runtime break.

**Document store** — one table, `documents`:

| Column | Type | Null | Default |
|---|---|---|---|
| `id` | uuid | NOT NULL | primary key |
| `user_id` | varchar(255) | **NOT NULL** | **none — never a server default** |
| `title` | varchar(500) | NOT NULL | none |
| `source_uri` | text | NULL | none |
| `object_uri` | text | **NOT NULL** | **none — never a server default** |
| `content_hash` | varchar(64) | NOT NULL | none |
| `document_kind` | varchar(64) | NOT NULL | `'generic'` |
| `status` | varchar(64) | NOT NULL | `'received'` |
| `jurisdiction` | varchar(255) | NULL | none |
| `contract_type` | varchar(255) | NULL | none |
| `parties` | jsonb | NOT NULL | `'[]'` |
| `metadata_` | jsonb | NOT NULL | `'{}'` |
| `created_at` | timestamptz | NOT NULL | now |
| `updated_at` | timestamptz | NOT NULL | now, advanced on update |

Constraint and indexes: unique `uq_documents_user_content_hash` on `(user_id, content_hash)` — **this is the
dedup key, and it is the pair, never the digest alone**; `ix_documents_user_id`; `ix_documents_kind` on
`document_kind`; `ix_documents_status` on `status`; `ix_documents_metadata_gin`, a GIN index on `metadata_`.

**Chunk store** — one table, `chunks`:

| Column | Type | Null | Default |
|---|---|---|---|
| `id` | uuid | NOT NULL | primary key |
| `document_id` | uuid | NOT NULL | foreign key to `documents.id`, `ON DELETE CASCADE` |
| `user_id` | varchar(255) | **NOT NULL** | **none — never a server default** |
| `chunk_index` | integer | NOT NULL | none |
| `chunk_kind` | varchar(64) | NOT NULL | `'generic'` |
| `content` | text | NOT NULL | none |
| `preamble` | text | NOT NULL | `''` |
| `clause_type` | varchar(128) | NULL | none |
| `page_no` | integer | NOT NULL | `0` |
| `embedding` | vector(*configured dimension*) | NULL | none |
| `metadata_` | jsonb | NOT NULL | `'{}'` |
| `custom_metadata` | jsonb | NOT NULL | `'{}'` |
| `quality_warnings` | jsonb | NOT NULL | `'[]'` |
| `graphiti_episode_id` | varchar(255) | NULL | none |
| `graphiti_verified` | boolean | NOT NULL | `false` |
| `search_text` | text | NOT NULL | **generated always as** `COALESCE(clause_type,'') \|\| ' ' \|\| COALESCE(preamble,'') \|\| ' ' \|\| COALESCE(content,'')` **stored** |
| `created_at` | timestamptz | NOT NULL | now |
| `updated_at` | timestamptz | **NOT NULL** | **now — new column, in the `CREATE TABLE`, never an `ALTER`** |

Constraint and indexes: unique `uq_chunks_document_chunk_index` on `(document_id, chunk_index)` — **the chunk
upsert key, named inside query text**; `ix_chunks_user_document` on `(user_id, document_id)`; `ix_chunks_kind` on
`chunk_kind`; `ix_chunks_metadata_gin`, GIN on `metadata_`; `ix_chunks_graphiti_verified`.

Three retrieval indexes, all on the chunk store, all required for the three-branch contract:

- `chunks_bm25_idx` — keyword relevance over `search_text`, English text configuration, `k1=1.2`, `b=0.75`.
  **Its name is read inside the query**, because the extension uses that index's corpus statistics.
- `chunks_embedding_idx` — approximate vector search over `embedding` with cosine distance.
- `chunks_search_text_trgm_idx` — GIN character-similarity index over `search_text`.

**Extensions the migration must create itself, not inherit.** Vector support, approximate-vector indexing, keyword
search and character similarity. This is not pedantry: the revision that declares the approximate-vector index
never creates the extension that provides it — only the sibling branch does, and that branch is
stamped-but-unrun — so the index survives today purely because the extension happens to be installed already. The
application role has been verified able to create the two that are missing.

**Must not be created:** the superseded search document and chunk tables, the clause table, and any second derived
text-search column or index over one. The embedding column's dimension must come from the single configured
source rather than a literal; unhardcoding it is change 1's, so change 0 should read it from one place rather than
repeat a number.

### Source-change order

Each step leaves the repository importable and the suite collecting.

1. Create the unified feature's constants module, carrying the retrieval constants plus the two identifier names
   the static gate asserts against, dropping the superseded index-name constant.
2. Relocate the chunking, rank-fusion and RAG helpers into the unified feature, leaving a re-export shim so both
   import paths resolve. The embedding client stays where it is — it is change 1's unification target.
3. Flip the unified feature's imports off the superseded feature. After this step it imports nothing from it
   except the embedding client, which is flagged rather than fixed.
4. Retarget the retrieval graph's fused search onto the unified repository, passing no clause filter and no
   verification filter, and clear the residual untyped attribute in the same module.
5. Add the static identifier gate. It must be **red before the retarget and green after** — that is what makes it
   a regression guard rather than a snapshot.
6. Delete the schema-bound twin, moving the graph-backed ask path onto the document query service unexposed, and
   rewrite the test surface **in the same commit**: the global conftest, the feature's integration test, and the
   relocated unit tests. Then drop the shim.
7. Delete the superseded ORM models and assert that no revision creates their tables outside the frozen history,
   and that the authoritative create-schema migration creates neither.
8. Delete the Celery ingest task, its re-export, its registration in the worker's include list, and the conftest
   line that stubs the module out — that stub would otherwise mask the deletion.
9. Add the schema-break gates: the static gate from step 5, and the real-database gate behind a marker so the
   default suite stays offline. Autogenerate comparison is **not usable** until change 0 rebuilds the database by
   upgrade rather than stamp, because it cannot distinguish a drifted model from migrations that never ran.
10. Add the chunk modification timestamp to the ORM, the conflict-resolution set and the row builder. No `ALTER`
    accompanies it; the column ships in change 0's `CREATE TABLE`.

### Coordination points

- Change 0 lands first: head merge, the authoritative create-schema migration, the task-registry rewrite, and
  model registration for the unified models only.
- Change 1 consumes the accepted ADR before implementing its persistence nodes.
- Change 3 owns the clause-reading tool stub; the handoff is recorded, and this change does not touch it.

### Proof status, stated honestly

Every verification in this change is source-level: imports, static scans, route enumeration, type and lint counts,
and the printed test summary. There is **no deferred migration proof**, because there is no migration — which is
a genuine simplification over the pre-database-probe plan. The one thing source-level proof cannot establish is
that the stores can be created and queried at all; only the real-database gate can, and nothing in the repository
has ever done it.

## Open Questions

- **Does the real-database gate run in continuous integration, or only on demand?** It needs a container and the
  four extensions. Gating it behind a marker keeps the default suite offline either way, so this is a scheduling
  and cost question rather than a design one, and it does not change the specs or the task breakdown.
- **Does anything outside the repository route to the deleted paths?** No client, test or generated API document
  inside the repository references them, and the router was never mounted, but deployment configuration and any
  gateway in front of the service were not inspected. A grep over deployment configuration before the deletion
  step closes it; it cannot change the design, only the deletion's blast radius.
- **Closed rather than left open:** the existing test-isolation capability was read and does **not** constrain the
  real-database gate — it governs one outbox unit test's use of a module proxy and says nothing about integration
  tests touching real services. The extension-permission question, which was the only open unknown that could have
  changed this change's content, was closed by probe: the application role can create both missing extensions, so
  retrieval ships three branches.
