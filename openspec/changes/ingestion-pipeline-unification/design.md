> Change class: **L** cross-cutting (multi-module, deployment surface, new external dependency, data-shape dependency).
> The proposal covers *why* and *what*; this covers *how*. Reference the proposal - do not restate it.

## Context

Two facts govern every decision below, and both were established by probing the live system rather than by reading
code.

**There are no tables.** The version table holds exactly one row, at the billing lineage's head. The entire
document, vector, and search branch — four revisions — is stamped but was never applied. The document, chunk,
search, clause, parent-document, durable-event, dead-letter, and memory-version tables are all confirmed absent,
with zero rows anywhere. The fifteen billing and audit tables genuinely exist. Someone stamped where they meant to
upgrade. Upgrade cannot repair it (the revisions are marked applied, so upgrade skips them) and downgrade fails
(its bodies drop tables that do not exist). This is the single cheapest moment in the project's life to settle the
embedding dimension, the chunk schema, and the normalisation convention, because nothing has to be preserved.

**The pipeline being promoted has never run, and neither has the path that would dispatch to it.** The dispatch
path carries four independent breaks stacked in series, each of which hides the next: the durable outbound event
table does not exist so the insert raises; the relay swallows the missing table behind a catch-all on both its
startup scan and its listener task, emitting one warning line each; the target task is registered only as a side
effect of one package initialiser importing another; and no worker or scheduler process exists in the deployment
at all, while the documented start command names an application module that does not exist. Fixing any one layer
in isolation produces no observable improvement. The first two layers belong to the foundation change. This change
owns the last two, and no verification here may be expressed as "a durable outbound event fires".

A third fact reshapes the checkpointer work. The checkpointer package is installed, but its driver has no libpq
binding, so the import raises and the module's fallback — aliasing the checkpointer type to a permissive
placeholder — is the **live** path. That fallback is currently the only reason the application boots on this
machine. Consequently the widely-cited defect in the module (calling a setup method on an unentered resource
manager) is currently **unreachable**; it becomes live the moment the driver is fixed. The two defects are real in
the opposite order from the one the plan assumed.

## Goals / Non-Goals

**Goals:**

- One embedding path, one configured dimension, one cache, one normalisation convention, task type declared on
  both sides.
- Structure-aware chunking for every document kind, with no silent content loss and no blocking parse.
- A checkpointer that is correct by construction: application-owned pool, real teardown, driver-shaped credentialed
  connection string, no placeholder alias, reference-only state within a stated size budget.
- Entity canonicalisation before the first knowledge-graph write.
- A deployment in which a process consumes the queue and the documented command starts it.
- One retrieval ranking path: fuse once, re-rank once, read the chunk records ingestion writes.
- The single-stage wrapper deleted and its three unique concerns absorbed by the multi-stage pipeline.

**Non-Goals** — each is a recorded gap, surfaced here rather than silently omitted:

- **Any database migration.** The foundation change owns all schema through one new migration on the merged head.
  This change ships no revision. Every requirement that reads or writes a table is gated on that migration.
- **Enabling the commented shared-graph and checkpointer wiring.** The user confirmed both are deliberate. They
  stay commented. See Decision 12.
- **Mounting the ingestion router**, for the same reason: a mounted route in front of an unprovisioned shared graph
  is a route that returns service-unavailable by design. The upload surface already exists on the documents router.
- **Mounting the retrieval router.** In scope means refactor and unify; mounting is gated on the request-scoped
  user identity fix, which belongs to the foundation change.
- **A shared vector-store singleton** (dropped): zero read sites exist, and retrieval is direct SQL against the
  lexical and vector extensions, so a framework vector-store object would be a **third** retrieval path.
- **"Refactor the retrieval code" as an umbrella item** (dropped): no acceptance criterion, and it restates this
  change. An unverifiable checkbox is forbidden by the workflow schema.
- **A second document parser** (dropped): the existing parser already owns parsing and was consolidated onto
  deliberately; a second one for the same job is subtraction disguised as addition.
- **Adopting a whole external agentic-retrieval architecture** (deferred): aspirational, no acceptance criterion,
  and it would balloon this change past reviewable size. The concrete pieces worth having — hybrid retrieval,
  re-ranking, agentic query rewriting — are already captured by the retrieval-ranking capability.
- **Collapsing the search tables into the unified document and chunk tables** (change 2), including the removal of
  the generated text-search column and its unread index.
- **Agent state shape, the tool registry, prompt adoption, and model-call middleware retries** (change 3). This
  change does not convert the pipeline state to a typed dictionary; see Decision 2.
- **Dropping the direct transformer dependencies** (unachievable as stated); see Decision 3.
- **The inconsistent vector width on the document-vectors table.** Recorded divergence, not migrated here.
- **The unused vector-dimension setting with zero readers.** Flagged for the foundation change's deletion sweep,
  not wired.
- **Moving re-ranking behind the queue.** The re-ranker's own note asks for this once query latency becomes
  visible; with a worker now existing there is somewhere to move it, but the move is follow-up work.
- **The phantom database function the advanced retrieval module calls.** This change fixes that module's
  unresolvable import; the function it depends on still exists in no migration. Recorded, not fixed.
- **Pre-existing routing gaps** where two task-name families do not match the routing pattern and fall to the
  default destination. Recorded, not fixed here.

## Decisions

### Decision 1 — Retry policy stays at input/output boundaries; model and tool retries belong to middleware

Retries already exist, using the retry library the sub-todo asks us to "add", and they are wrong in policy rather
than absent: the retry predicate is the base exception type, the wait is zero, the re-raise flag is dead because
the loop is wrapped in a catch-all that re-wraps every distinct failure into one opaque transient type — which is
not the framework's base exception, so the pipeline's own degradation branches **can never fire** for a wrapped
call. We fix the policy in place: named transient types, growing wait, original exception preserved and chained.

*Alternatives considered.* (a) Add a second retry mechanism as the sub-todo literally reads — rejected: two
policies at one boundary is worse than one wrong policy. (b) Move all retries into model-call middleware now, as
the reference documentation prescribes — rejected: that is change 3's seam, and the boundary being fixed here wraps
database, cache, object-store, and graph calls that middleware never sees. (c) Remove retries entirely and rely on
checkpoint replay — rejected: replay's recovery unit is the stage, so a transient rate-limit would re-run an entire
stage including its completed side effects.

The resulting layering, binding on change 3: **retry wrappers stay at input/output client boundaries with named
types and growing waits; model and tool retries are owned by middleware; replay safety is owned by idempotency
keys, not by attempt counters.** A node-local attempt counter resets on replay, so the retry budget is silently
multiplied once a checkpointer exists — which is why the policy fix lands before the checkpointer goes live.

### Decision 2 — The pipeline state stays a validated model; this change only shrinks it

The reference corpus states that custom state schemas must be typed dictionaries, but that statement is scoped to
the prebuilt agent constructor's state schema, and this pipeline uses a bare graph builder. The evidence for the
bare builder is genuinely absent, and a superseded in-repo plan prescribed the validated-model shape as house
style. So this change **does not convert** the state type.

*Alternatives considered.* (a) Convert now to match the documented direction — rejected: it would be a guess
presented as a standard, and it is change 3's decision. (b) Leave the state alone entirely — rejected: two of its
channels are serialisation hazards that must be removed before the first checkpoint is written.

What this change does instead: shrink the channels, remove the arbitrary-types permission (the single biggest
obstacle to a later conversion), and add **no** new model-only affordances — no validators, no computed fields, no
arbitrary types. The state is left convertible. Change 3 should settle the bare-builder question from the graph
builder's own documentation, not from another pass over the repo corpus.

### Decision 3 — The "drop the transformer dependency" item is unachievable, and is recorded as such

Its premise is false. One transformer package supplies the cross-encoding model that **is** this change's
re-ranker; the other supplies the token counter that **is** this change's chunker input. Both are declared direct
dependencies. Dropping either would break two other in-scope items in this same change.

*Alternatives considered.* (a) Drop the packages and find framework replacements — rejected: there is no
equivalent cross-encoder in the framework surface, and the chunker's counter parameter expects that specific
interface. (b) Drop the item silently — rejected: a backlog item that quietly vanishes reads as done.

**What survives is real and is kept.** The token counter is acquired uncached, synchronously, with a first-use disk
or network load, on every call — that is fixed. And the counter in force is **not the embedding model's counter**:
chunks are budgeted by one model's token count and then embedded by a different provider, so the token bound is
enforced against the wrong counter. That mismatch is a genuine correctness gap in the chunk-size guarantee, and it
is the finding the item would have surfaced by accident. It is recorded in the chunking capability as an explicit
requirement: either the counter matches the embedding model, or the divergence and its safety margin are stated.

### Decision 4 — The framework's cache-backed embedding wrapper is rejected; the shared cache satisfies the intent

The reference documentation names the exact defect (every batch call hits the API) and prescribes the framework's
cache-backed embeddings wrapper over a local file store.

*Alternatives considered and rejected.* (a) Adopt the wrapper as prescribed — rejected on two independent grounds:
the class is only importable from the version-zero compatibility shim, and the project's own import rule forbids
legacy import paths; and the prescribed backing store is per-container, so it is useless behind a scaled service
and would silently give each replica its own cache. It also requires the embeddings object to expose a model
attribute. (b) Add the wrapper over a shared-cache-backed store — rejected: it wraps a mechanism we already have in
order to add a legacy import and a second cache tier.

**Chosen:** satisfy the intent with the shared cache keyed by a digest of text, model, and task type. It is
cross-process, it is already the project's cache substrate, and it already exists in this exact shape twice in the
codebase — those two implementations collapse into one. The item is answered on the record, not dropped.

### Decision 5 — The queue item is a deployment change, not a configuration line, and it is re-ranked

The briefed mechanism was wrong. The task **is** registered today, transitively, because one package initialiser
imports it and importing any listed sibling imports that initialiser first. The omission from the explicit module
list is therefore a **latent fragility**, and it becomes live precisely when the foundation change tidies that
initialiser — a genuine cross-change hazard where one change edits the file that silently guarantees another's
dispatch.

The real causes rank ahead of it: **no worker and no scheduler process exist** in the deployment, and the
documented start command names a module that does not exist. Order: add the processes, fix the command to match
them exactly so they cannot drift, then make registration explicit and add shared task-name constants.

*Alternatives considered.* (a) Add the module list entries only, as briefed — rejected: it changes nothing
observable, because nothing consumes the queue. (b) Run ingestion in the API process — rejected: minutes-long work
in a request worker. (c) Defer the deployment change to an operations task — rejected: the queue item cannot be
verified by any code-level check, so without the process the requirement has no proof.

**Sequencing note that a reviewer will otherwise flag as a regression:** if the foundation change lands first, as
its ordering requires, there is a window in which explicit registration has not yet landed. Accept it — nothing
consumes the queue in that window anyway, so it has no observable effect.

### Decision 6 — The claim that the lexical and fusion code "already works" is downgraded to "written and tuned, never executed"

The justification for pulling the retrieval module into scope was that its lexical and fusion implementations
already work, so rebuilding them elsewhere would duplicate working code. The first half of that is now known to be
false: the router is unmounted so no request reaches them, the in-database variant queries a table no migration
creates, the fuzzy extension is not installed so its index has never existed and that branch has **never run**,
and the tests patch the repository and use a mock session so nothing touches a database.

*Alternatives considered.* (a) Withdraw the module from scope — rejected: freezing it also freezes the retrieval
graph, whose only caller lives there, and the retrieval graph is squarely this change's. (b) Rebuild lexical and
fusion in a new place — rejected: that is exactly the duplication the reversal was made to avoid.

**Chosen:** keep the module in scope, keep the code as the foundation, and downgrade the claim. Concretely: the
fuzzy branch is treated as a **new** branch to bring up (its extension and index must be created, not assumed), and
extension availability is a **recurring per-environment precondition** checked against the database the application
connects to — not against a container image, and not once. A green lint, type, and test run must never be allowed
to stand in for "lexical search works"; nothing in the test suite touches the database.

The precondition closes favourably today: the lexical extension is available at version 1.3.0 on the managed
instance the application actually connects to, and not yet installed. The vector and vector-index extensions are
installed; the fuzzy, accent-folding, and identifier extensions are available and not installed. The lexical index
access method is consequently **absent** today and must appear after the extension is created. The compose database
has never been up in this working copy — nothing listens on its port — so any check aimed at the container image
answers a question nobody asked.

### Decision 7 — Splitter: the parser's own structure-aware chunker. Framework vector-store classes: neither

The backlog asks which text splitter to use, and whether to adopt one of two framework vector-store classes. Both
are answered here rather than as code tasks.

**Splitter: the document parser's own structure-aware chunker.** It already exists and is already correctly
configured — token-bounded, peer-merging, heading-path-contextualised, keyed to the parsed document structure. It is
reachable only for generic documents. For legal documents the live path splits on a blank-line pattern, which
discards the heading hierarchy the parser extracted, the token budget, and the peer merging — and then truncates to
two hundred sections, **silently dropping everything after that with no warning**, while the function's only
quality warning fires on the opposite condition. For a fifty-page contract that is most of the document. This is
data loss, not a quality preference.

*Alternatives considered.* (a) A recursive character splitter — rejected: it cannot see heading structure, which is
the whole point for legal documents. (b) Keep the pattern split and add a clause-aware pass — rejected: it rebuilds
what the parser already produced, from a lossier input.

**Framework vector-store classes: neither.** Retrieval is direct SQL against the vector and lexical extensions.
Introducing a framework vector-store object would create a third retrieval path with its own dimension, filter, and
identifier conventions. The clause-boundary awareness the structure-aware chunker lacks is added on top of it, not
in place of it.

### Decision 8 — Durability: persist while the next stage executes

*Alternatives considered.* (a) Persist only at completion — rejected outright: the documentation is explicit that
this mode cannot recover from a failure occurring mid-execution, which is precisely the failure this change exists
to fix; today a crash at the ninth stage replays the first eight. (b) Persist synchronously before continuing —
defensible, and it is the right choice for the knowledge-graph stage specifically, whose writes are the
non-idempotent ones. (c) **Chosen:** persist while the next stage executes. Ingestion stages are long enough (model
calls, parses) that the write is fully hidden, and the residual crash window is one stage boundary.

### Decision 9 — The fuzzy branch is retained, and its extension and index are requested from the foundation change

It is one of three fusion branches, so dropping it silently degrades fusion quality; and its extension is not
installed and its index has never existed, so it cannot simply be harvested.

*Alternatives considered.* (a) Drop the branch — permitted by the reversal only if recorded, and it would make
fusion two-branch. Rejected: fuzzy matching earns its place on legal text, where party names and defined terms are
misspelled and reformatted constantly. (b) Bring it up here with its own migration — rejected: this change ships no
schema. **Chosen:** retain the branch, request the extension and index in the foundation change's single migration,
and require the branch to be omitted observably where either is absent.

### Decision 10 — Persistence writes chunk records, never clause records

The schema contract is accepted before this change implements persistence — the work-order sequence bends exactly
once, for the schema only. The pipeline's persistence stages target the unified document and chunk records from the
start, and write no clause records, no parent-document records, and no relational entity or relationship records as
retrieval truth. Extracted entities and relationships go to the knowledge graph.

*Alternatives considered.* (a) Write clause records now and migrate later, following the work order strictly —
rejected: the clause table does not exist and never has, so writing to it first would preserve nothing and would
roughly double the following change with a migration for zero rows. (b) Write both — rejected: two retrieval truths
is the defect being removed.

### Decision 11 — This change ships no schema, and fails closed when the schema is absent

All schema arrives through the foundation change: merge the two revision heads, then **one** new migration that
creates the target schema outright. Editing the unapplied revisions in place was explicitly rejected (we cannot
prove no other environment applied them), as was a full rebaseline (the fifteen genuinely-existing billing tables
would need hand reconciliation against a rewritten root).

*Alternatives considered.* (a) Ship the vector-column definitions with their own migration here — rejected: two
changes writing schema for the same tables is how the current stamped-but-unapplied situation was created. (b)
Assume the schema exists — rejected: today every count-shaped check against those tables returns a
relation-does-not-exist error, not a zero.

Accepted cost, on the record: the revision chain permanently reads as a lie — three revisions stay stamped while
creating nothing. Because there is no data, the vector-width work here is a **column definition**, not a type
migration: no widening, no index drops, nothing to preserve.

### Decision 12 — The commented shared-graph and checkpointer wiring stays commented; ingestion runs in the worker

The user confirmed that leaving the shared pipeline graph and the checkpointer unwired in the application lifespan
was deliberate. This **overturns** the assumption the plan carried, which had treated it as a regression by analogy
with a genuinely live defect on a mounted router. The consequence is structural, not cosmetic: the plan's wiring
step cannot be performed, and the end-to-end acceptance check that step enabled is unavailable inside this change.

*Alternatives considered.* (a) Uncomment the wiring anyway because the pipeline needs it — rejected: it contradicts
a locked decision. (b) Introduce a feature flag defaulting to enabled — rejected: that is the same thing with extra
steps. (c) Drop the pipeline promotion — rejected: it contradicts the locked decision that the multi-stage pipeline
becomes the real one.

**Chosen, and it resolves the tension rather than splitting it:** ingestion runs in the **queue worker process**,
which never executes the application lifespan and therefore never had access to shared application state in the
first place. The build-once requirement applies per worker process, not to application state. The synchronous HTTP
ingestion surface that reads the shared graph stays unprovisioned and must **fail closed** with a typed
service-unavailable error — which is now the primary justification for that fix rather than a side effect of
restoring wiring. The router is not mounted, so no service-unavailable surface ships.

**Therefore the proofs for the checkpointer and the shared-graph construction are import-level and type-level
only.** Commented code cannot be linted, type-checked, or executed, so it rots. Every task touching it proves
correctness *by construction* — imports resolve, types check, the constructor is exercised by a unit test — and
never by running the graph through the application. Say this in the task text too, or an implementer will reach for
an integration test that cannot exist. The phrase "deliberate at that time" is noted: nothing here should make
re-enabling the wiring harder later.

### Decision 13 — The driver fix and the deletion of the placeholder alias are one commit

The checkpointer's placeholder alias is the **live** path and currently the only reason the application boots on
this machine. Installing a working driver binding and deleting the alias must therefore land **together**;
splitting them produces a commit that does not boot, violating this change's own per-step rule.

*Alternatives considered.* (a) Rely on a system-provided client library instead of the binary wheel — rejected: not
reproducible across the container image and this working copy, and the connection-pool package already being
installed is strong evidence the pooled shape was the original intent. (b) Keep the alias as a defensive fallback —
rejected: it converts a hard dependency failure into an attribute error at the first agent request, which is worse
than a boot failure because a startup-only check passes.

Restated boot claim, since the earlier version was wrong in the other direction: as things stand today,
uncommenting the wiring would log **one warning**, leave the shared slot absent, and then raise an attribute error
at the first agent request. Not a boot crash — a silent degradation that becomes a crash later, in the same
invisible-failure register and materially worse, because continuous integration that only checks startup passes.

### Decision 14 — The checkpointer gets its own connection-string accessor; there are three flavours, not two

Neither existing option works. The raw configured URL carries **no password** — the relational engine works only
because its accessor repairs the URL, injecting the password and stripping the transport parameters its driver
rejects. But that accessor returns the relational engine's **dialect alias**, which the checkpointer driver cannot
parse. One is unauthenticated, the other unparseable.

The system needs **three** URL flavours: the relational engine's dialect-aliased form; a plain client-library
form, which the checkpointer and the durable-event relay's listener both need and which must **retain** the
transport-security parameter the other driver rejects; and the memory subsystem's form. The dialect alias is
currently stripped in at least two separate places, and a third component reads the raw passwordless URL — the same
class of defect in three locations.

*Alternatives considered.* (a) Reuse the relational engine's accessor — rejected: unparseable scheme. (b) Read the
configured URL raw — rejected: unauthenticated. (c) Repair the string at each call site — rejected: that is the
current state, and it is why three components disagree.

**Chosen:** exactly one accessor per flavour, with credential injection and scheme repair living only there, so no
consumer can obtain an unusable URL. The durable fix — that the repair belongs in configuration or in a single
accessor set rather than in call sites — is the foundation change's to complete; this change adds the
client-library-flavour accessor it needs and requires no consumer to transform another flavour's string.

### Decision 15 — The batch embedding implementation survives as batch-only and must not become a second live path

It is a carve-out: it stays because the local-folder batch ingester imports its batch function. It is fixed here
(configured dimension, no placeholder vectors) but it is **not** a live path, and no request or ingestion stage may
reach it.

*Alternatives considered.* (a) Delete it and retarget the batch ingester onto the unified path — defensible and
cheaper long-term, but the batch ingester is a genuinely different use case with its own database-cleaning and
pipeline-construction entry points, and retargeting it is not this change's work. (b) Leave it unfixed since it is
not live — rejected: it is the module that returns a 1536-dimension contract against 768-wide columns, and its
placeholder vectors propagate the wrong width silently.

### Decision 16 — The archived node-failure-pattern capability is harvested, not delta'd

An archived change shipped a capability binding how pipeline nodes signal failure, and it explains the existing
split between guard-clause failure records and framework result values. It is **not** present in the live
capability directory — the twenty live capabilities were enumerated and none matches — so there is no live
requirement block to copy.

*Alternatives considered.* (a) Create a new capability under the same name — rejected: it would collide with the
archived one at archive time and would fork one contract into two. (b) Ignore it — rejected: its contract is
exactly the failure-signalling behaviour this change's short-circuit edge depends on.

**Chosen:** harvest its text into the pipeline capability's failure requirement, preserving the existing contract
(a failed stage yields a failure record in the failure channel, logged at the boundary) while adding the two things
it did not bind: that the record must be serialisable, and that a recoverable-failure handler must not destroy the
original diagnostic.

### Decision 17 — The exception-taxonomy capability is extended, not modified

Two of this change's concerns are exception-taxonomy changes on an existing capability: embedding failures must
raise instead of substituting, and retry boundaries must stop collapsing every distinct failure into one opaque
type. The existing capability binds neither concern, so these are **additions**, and the delta uses the added
operation rather than the modified one — which also avoids copying a large requirement block that would silently
lose detail if copied imperfectly.

*Alternatives considered.* (a) Use the modified operation as the plan's mapping proposed — rejected: nothing in the
existing text is being changed, and the modified operation requires copying the entire original block. (b) Put these
requirements in a new capability — rejected: the existing one is exactly the right home, and reusing it is the
workflow's stated preference.

**Attribution note, recorded before editing because it cannot be reconstructed afterwards.** Two of the six
pre-existing validation failures are capabilities this change touches or was proposed to touch. Their failure
output today, verbatim:

- `spec/typed-exception-handling`: `[ERROR] file: Spec must have a Purpose section. Missing required sections.
  Expected headers: "## Purpose" and "## Requirements".` — the live file is delta-shaped and has no purpose
  section. Unrelated to any delta added here.
- `spec/transactional-outbox`: six errors, `requirements[0..5]: Requirement "<name>" must contain SHALL or MUST`.
  Unrelated to any delta added here.

### Decision 18 — The durable-event capability is left alone; the task-name constant lives with the worker deployment

The shared task-name constant changes how a dispatched name is resolved, which is adjacent to the durable-event
contract. But that capability is one of the six pre-existing validation failures, and delta'ing an already-failing
spec makes attribution of any new failure ambiguous.

*Alternatives considered.* (a) Delta it — rejected on the attribution ground above, and because the durable-event
contract itself does not change: an event still records, relays, and dispatches exactly as specified. (b) Delta the
configuration-validation capability for the dimension work — rejected for the same "decide once, do not delta both"
reason; the dimension contract lives in the embedding capability, which is where its consumers read it.

**Chosen:** the task-name constant requirement lives in the worker-deployment capability.

## Risks / Trade-offs

**[The promoted modules have zero covering tests, so "it still works" is unfalsifiable]** → No test references the
pipeline package, the single-stage wrapper, or the dispatch task, and no covering test exists for any of the seven
stage factories, the graph builder, or the task entry point. The evidence that stands in today is: lint, types,
structural scan, and a passing suite that touches none of this code. That proves the repository *imports*, and
nothing more. **Mitigation:** twelve mandatory new tests are the regression net being built, placed exactly where a
defect is invisible to lint and types — a blocking call, a silent truncation, a placeholder-vector substitution, a
swallowed exception, a serialisation-size regression, an unrecoverable entity duplicate. No such step may ship on a
lint-only check.

**[The coverage gate makes a green suite exit non-zero, so the exit code is a lie]** → The gate demands eighty per
cent against current coverage near eighteen. **Mitigation:** every check compares the suite's summary counts and
states the expected passing count, so a silently lost test shows up as a count that did not rise. The specific
danger is an implementer wiring a hook on the exit code, concluding the suite is broken, and "fixing" it by lowering
the gate.

**[Entity canonicalisation is the only irreversible step in the change]** → Duplicate party nodes cannot be
separated after the fact, because the disambiguating context is the extraction already discarded. **Mitigation:**
canonicalisation lands before any knowledge-graph write goes live, and its unit test is the most load-bearing new
test in the change. Secondary mitigation: there are three graph write sites today; **audit all three by reading, not
by pattern search** — a missed site poisons the graph exactly as thoroughly as no canonicalisation at all.

**[The end-to-end acceptance check does not exist inside this change]** → Because the shared wiring stays commented
by decision, there is no path from a mounted route through a provisioned graph to a persisted chunk row inside this
change. **Mitigation:** the acceptance evidence is decomposed — construction-level proofs for the graph and
checkpointer, a worker-interrogation proof for the queue consumer, unit-level proofs for every correctness fix, and
a checkpoint round-trip against a reachable database for resumability. State plainly in the task list that the
single upload-to-chunks check is **not** available here, so nobody writes it as a task and then deletes it when it
cannot pass.

**[Deleting the live single-stage path is the one step that is not independently revertible]** → It removes the
currently reachable implementation. **Mitigation:** it is last, every prerequisite is separately proven, and the
multi-stage replacement must be exercised by the dispatch task under test before the deletion. If the replacement
cannot be exercised, **the deletion must not be attempted** — the change ships with the wrapper still present and
the promotion completed in the following change. State that fallback in the task list.

**[The foundation change edits the one file that silently guarantees this change's dispatch]** → Task registration
holds today only as an import side effect of the package initialiser that the foundation change is about to tidy.
**Mitigation:** the registration check runs **after** that edit, not only before; and adding the explicit module
entries removes the dependency entirely. There is a window where dispatch is broken; nothing consumes the queue in
that window, so it has no observable effect.

**[Every table-shaped check is currently a relation-does-not-exist error, not a zero]** → No document, chunk,
search, clause, durable-event, or memory table exists. **Mitigation:** every requirement that reads or writes a
table is explicitly gated on the foundation change's single migration, and the pipeline is required to fail closed
with a diagnostic naming the missing schema rather than leaving a document in a non-terminal status.

**[Changing the configured embedding dimension later is a re-embedding campaign, and tracking it from settings makes
it look like a knob]** → The vector type's width modifier is not widenable in place; every vector index on the column
must be dropped first, and the type change fails while any row holds a different width. **Mitigation:** the column
definition tracks the setting but the setting is documented as read at process start, and the capability requires
that a configured width differing from stored vectors refuses new writes and reports that re-embedding is required.
Today this risk is purely forward-looking — there is no data — which is why this is the cheapest moment in the
project's life to settle the value.

**[Changing the pipeline state shape makes any checkpoint written under the old shape unreadable]** → **Mitigation:**
there is no legacy checkpoint data, because the checkpointer's setup has never been reached and therefore its tables
cannot exist. This is a free window that **closes permanently** the moment the checkpointer is attached. If the
ordering slips and attachment precedes the state shrink, the mitigation evaporates and a migration becomes necessary.

**[The lexical extension's availability is controlled by the vendor, not by this repository]** → It is available and
uninstalled on the managed instance today, and there is **no fallback**: the two alternative lexical extensions are
not available at all. Creating an extension conditionally does not protect against a missing one — it only suppresses
"already exists", so a missing extension aborts the whole migration, taking the durable-event and billing schema with
it. **Mitigation:** availability is a recurring per-environment precondition, and the lexical branch is required to be
omitted observably rather than to error where the extension is absent.

**[A swallowing error handler behind the ingestion service could turn a failed ingestion into a success response]** →
The service logs an exception at one point and it was not established whether it re-raises. **Mitigation:** the task
that touches that file must read the handler, not pattern-match it. An unmounted swallowing handler is invisible; the
same handler behind a live surface is a correctness bug on shipped surface.

**[Scope creep by reading "the retrieval module is in scope" as "collapse the schema now"]** → It is not: the
collapse is the following change and mounting is gated elsewhere. **Mitigation:** the retrieval work here is about
one fusion implementation, index-name constants, re-ranking, and extension preconditions — **not** about table names.
If a task starts editing retrieval column definitions, it has left this change.

**[Seven new capability directories, and the four-hashtag scenario trap fails silently]** → Three hashtags or bullets
drop the scenario with no error. **Mitigation:** a pattern search for scenario headers with one to three hashtags
across the change's spec files must return zero, run as the final check before validation; it returns zero today.

## Migration Plan

1. **Nothing to back-fill, by accident.** No document, chunk, or search rows exist, and no checkpoint tables exist,
   so the embedding dimension, the chunk shape, the normalisation convention, and the pipeline state shape are all
   settled without data migration. This reasoning is not obvious to a later reader and is recorded here deliberately.
2. **The foundation change lands first**: merge the revision heads, then one new migration creating the target
   schema — the document and chunk tables with the additional timestamp column, the durable-event and dead-letter
   tables, the vector index, and the lexical, fuzzy, accent-folding, and identifier extensions.
3. **Install a working database driver binding and delete the placeholder alias in the same commit.** This is a hard
   precondition for any checkpointer work, and nothing else in the change depends on it, so the correctness fixes,
   the seam unifications, and most of the retarget proceed in parallel with it.
4. **Correctness fixes before seams before substrate.** The diagnostic-logging fix precedes anything that changes a
   dimension, because until it is fixed every dimension mismatch is an error rather than a warning and six tests stay
   red — no later step has a clean baseline. The dimension-contract fix precedes the column definitions.
5. **The state shrink precedes the checkpointer attachment, absolutely.** Once a checkpoint is written, the shape of
   what was written is history.
6. **Canonicalisation precedes any knowledge-graph write going live, absolutely.**
7. **The single-stage wrapper is deleted last**, and only after the multi-stage replacement is exercised.
8. **Rollback**: every step through the substrate band is independently revertible. The deletion is not; if it must
   be reverted, the wrapper and its state type return together with the router remaining unmounted.

## Open Questions

1. **Does ingestion get its own queue, or share the default one?** Recommendation is a dedicated queue with its own
   concurrency, because minutes-long model work will otherwise starve sub-second billing and transactional-email
   tasks. But the configuration forbids creating queues implicitly, so the queue set is fixed and this is a
   deliberate operational decision with a cost, and **no locked decision covers it**. This changes the task
   breakdown, so per the workflow schema it must be **asked, not guessed** — it must be answered before the task
   list is written. The capability states the behaviour (long work must not starve latency-sensitive work) and
   leaves the topology to this answer.
2. **Does the bare graph builder still accept a validated-model state type on the installed framework version?**
   Inherited and deliberately left open. It does not block this change, which only shrinks channels, but it blocks
   change 3's conversion decision. Resolve from the graph builder's own documentation, not from another pass over
   the repository corpus, which is scoped to the prebuilt agent constructor.
3. **Does the framework validate a fan-out payload against the state schema?** The per-chunk fan-out constructs a
   plain mapping carrying keys the state type forbids as extras, while the graph is compiled against that state
   type. If the framework validates, the fan-out is rejected and the fold hits it immediately. Resolved by the
   checkpoint round-trip check, which invokes the graph once — so it is answered by work already planned, but it is
   recorded because the answer changes the fold's shape.
