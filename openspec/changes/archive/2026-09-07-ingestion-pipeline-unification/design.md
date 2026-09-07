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
  stay commented, and D17 forbids re-enabling them here. The consequence is scoped deliberately: this change does
  **not** wire the checkpointer into the application lifespan, does **not** make the FastAPI application the owner of
  a checkpointer connection pool, and therefore ships **no** proof that runs through the lifespan. Every checkpointer
  proof here is import-level, type-level, or a unit test over a construction the test itself owns. See Decision 12
  and Coordination point 3.
- **The fused-retrieval contract, the single retrieval source of truth, and extension preconditions** — all owned by
  change 2's `document-retrieval-schema`. This change's retrieval capability contains no fusion requirement, no
  single-source requirement, and no degrade-and-continue behaviour for a missing database extension. See
  Coordination point 1.
- **The read-site fail-closed contract for an unprovisioned shared checkpointer** — change 3's
  `agent-runtime-resilience`. This change provisions the checkpointer honestly; the 503 at the read site is change
  3's step 1. See Coordination point 2.
- **The database URL accessor set** — change 0's `infrastructure-client-access`. This change is a consumer. See
  Coordination point 4.
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

## Coordination points

Five boundaries where this change and a sibling change touch the same code. Each was a review finding: two changes
independently specifying one code path is invisible from inside either one, so each is recorded on **both** sides.
Ownership is settled, not negotiable at implementation time.

1. **A missing lexical or fuzzy extension fails loudly. Change 2 owns it.** Change 2's
   `document-retrieval-schema` requires that a declared retrieval mode whose required extension or index is absent
   causes provisioning to **fail loudly**, and forbids silently serving a fused result from fewer modes than the
   system declares. This change originally required the opposite — omit the branch, continue fusion, report the
   omission — over the same code path. Ruled in change 2's favour, for two reasons that also generalise: change 0
   creates all four extensions explicitly (D14.4), so a missing extension at runtime means the migration did not run,
   which is a deployment error rather than a runtime condition; and degrade-and-continue is the exact pattern that
   left this repository's outbox permanently dead behind two warning lines. Consequently this change **deleted** its
   degrade-and-continue requirements for both the lexical and the fuzzy branch, and **deleted** its duplicated fusion
   and single-retrieval-source requirements. What remains here is re-ranking, index identity, and extraction
   ordering. Change 2 must not weaken its scenario to meet this change halfway.
   *One deliberate asymmetry, stated so it does not read as a contradiction:* the **re-ranker model** being
   unloadable still degrades to the fused order and reports on the health surface. A model download failure is a
   recoverable runtime condition whose degraded output is still a correct ranking; an absent extension is a
   deployment error. The retrieval capability states this distinction in the requirement body.

2. **The fail-closed 503 for an unprovisioned checkpointer is change 3's.** D17 names
   `features/agent_saul/dependencies.py:45` reading `app.state.langgraph_checkpointer` unguarded as the defect, and
   makes it the primary justification for change 3's step 1. Split: **this change provisions** the checkpointer
   (`dispositions.md` item 138 residue a) and guarantees it never returns an absent value from a function typed to
   return a saver; **change 3 owns the read site** and its typed service-unavailable response. This change's
   `langgraph-checkpointing` capability no longer carries a consumer-side requirement.

3. **The application lifespan wiring stays commented, so pool ownership is the worker's.** D17 forbids re-enabling
   the two deliberately commented blocks and requires proofs there to be import-level and type-level only. The
   original requirement — "the application SHALL create and own the connection pool … shutdown SHALL close that
   pool", with an observable-closure scenario keyed to application shutdown — could only be satisfied by writing the
   forbidden lifespan wiring. Re-scoped to what is provable by construction: the **constructing process** owns and
   closes its pool, in this change that process is the queue worker, and the requirement now states explicitly that
   the disabled application construction stays disabled.

4. **The database URL accessor set is change 0's `infrastructure-client-access`.** This change deleted its own
   "each flavour has exactly one accessor" requirement and became a consumer. Corrected fact, replacing what
   Decision 14 originally claimed: there are **two** flavours, not three — SQLAlchemy-plus-asyncpg, and plain
   libpq/psycopg. The psycopg flavour exists **because of this change's checkpointer**, which can parse neither the
   raw configured URL (no password) nor the relational engine's dialect-aliased form. See Decision 14.

5. **The lexical index name is part of the query contract, and change 0's migration must create it by exact name.**
   The lexical extension's `to_bm25query` has a two-argument overload taking the index name as a **literal
   argument**, and the repository uses that overload. An index of the correct shape under a different name does not
   satisfy the SQL. So the constant this change pins is the same string change 0's migration must use. This change
   owns hoisting the literal to the constant; **change 0 owns creating the index**, and until it does, the lexical
   branch cannot run at all — there is currently no `bm25` index anywhere in the database
   (`findings-database.md` §10).


## Decisions

### Decision 1 — Retry policy stays at input/output boundaries; model and tool retries belong to middleware

Retries already exist, using the retry library the sub-todo asks us to "add", and they are wrong in policy rather
than absent: the retry predicate is the base exception type, the wait is zero, the re-raise flag is dead because
the loop is wrapped in a catch-all that re-wraps every distinct failure into one opaque transient type — which is
not the framework's base exception, so the pipeline's own degradation branches **can never fire** for a wrapped
call. We fix the policy in place: named transient types, growing wait, and the original exception reachable as the
chained cause.

**A correction that the original wording got wrong, and it is the kind of error that ships broken code.** The first
version of this decision said the fix was to chain via `raise … from exc` "so a caller's existing degradation branch
still matches". It does not. Chaining sets `__cause__`; it does **not** change the type of the exception raised, so
an `except LangChainException` around a boundary that raises `TransientExternalError` will not match no matter how it
is chained — and `nodes.py:236` is exactly that `except`. Chaining and type-preservation are two different
properties, and only one of them was actually being delivered.

Two coherent contracts existed. Either the boundary re-raises the **original type** and attaches the retry context as
a note, leaving callers untouched; or the boundary raises **one typed transient failure** chained to the original, and
the callers are converted to catch it. **Chosen: the second.** It matches this decision's own direction — a single
named transient type at the boundary is what makes the retryable set nameable in the first place — and it makes the
conversion of callers an explicit, greppable step rather than an invisible assumption. The cost is honest: converting
callers is work, and a caller that is missed is a degradation branch that silently stops firing. That is why the
capability requires the caller inspection as a scenario in its own right, not as a side effect.

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

**Resolution (B4, measured 2026-08-23).** Both halves of "what survives" are now settled, and one sentence of this
decision's own premise is withdrawn.

*The cache.* The counter is loaded once per process. The load moved out of the public accessor into a memoised
loader keyed on a **resolved** model id, with the accessor normalising the default before delegating — a default
argument is not part of a memoisation key, so decorating the accessor directly would have given a call that omits
the argument and a call that passes the identical default value two entries and two loads. The cache is bounded
rather than unbounded because its key is caller-supplied. The log line moved inside the loader, so a line in the log
now means a load actually happened; emitted per call, as before, it was false for every call after the first.

*The counter mismatch: the divergence is **stated**, not closed* — the second of the two options above. Matching is
not available on the terms this project can meet: the embedding provider is Gemini (`gemini-embedding-001` as passed
by `embedder.py`, against `gemini-embedding-2-preview` in configuration — a second divergence that is **B1's**, not
this task's), and the installed provider SDK exposes token counting only as a remote call. Matching the counters
would therefore put a network round trip inside every chunk-boundary decision, inside a synchronous chunker. It
would also change chunk boundaries, which is a corpus change, not a seam change, and so outside Band B.

The margin, derived entirely from this repository's own two constants rather than from any external claim:

| Quantity | Value | Source |
|---|---|---|
| Chunk budget, enforced | 512 tokens, counted by a WordPiece counter | `IngestionConfig.max_tokens` |
| Bound that actually applies downstream | 8192 **characters**, applied by silent truncation | `embedder.py:146,210` — `_MAX_INPUT_TOKENS * 4` |
| Budget expressed in the downstream unit | ~2048 characters, at the ~4 characters-per-token density that same guard assumes | derived |
| **Headroom** | **~4x** | 8192 / 2048 |

Two things that table makes visible and that were not on the record before. First, the downstream bound is a
**character** bound, not a token bound — the provider's token limit is only ever approximated here by a
four-characters-per-token rule of thumb, so the two guards are not merely counted by different tokenizers, they are
counted in different units. Second, the margin degrades in exactly one direction: WordPiece maps an unsegmentable
run of up to 100 characters onto a *single* unknown token, so the counter undercounts without bound on base64
blobs, hex digests and long URL path segments, and a chunk of 512 such tokens can exceed 8192 characters and lose
its tail to that truncation with no diagnostic. The 4x is a statistical margin, not a proof. Enforcement belongs on
the embedding side, where the constant already lives, and therefore to **B1**; a chunker cannot see which provider
will embed what it emits.

*Correction to this decision's premise.* "Both are declared direct dependencies" is **false**, and the error makes
the conclusion stronger rather than weaker. Only the sentence-transformer package is declared (`pyproject.toml:51`);
`transformers` is declared nowhere and arrives transitively by four independent paths — through the
sentence-transformer package, through `docling` → `docling-ibm-models`, through `docling-core[chunking]`, and
through `headroom-ai[all]`. So the tokenizer half cannot be dropped by deleting a declaration, because there is no
declaration to delete: `transformers` would still be installed by the parser this change is built on (Decision 7).
Combined with Decision 19 settling that the sentence-transformer package stays for the cross-encoder re-ranker
— verified directly: `retrieval_kb/reranker.py:8` imports `CrossEncoder` at module scope — the item is
unachievable in **both** halves, for two different reasons, neither of which is the one originally given.

That leaves a hazard worth its own task, which no task in `tasks.md` currently owns: `src/` imports `transformers`
directly at `rag/document_processing/chunker.py:55` while `pyproject.toml` does not declare it. The import is
protected by nothing but the resolution graph of four other packages. It should either be declared explicitly or
not imported directly; it is recorded here rather than fixed under B4 because `pyproject.toml` and the lock file are
contended by C1.

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
fuzzy branch is treated as a **new** branch to bring up (its extension and index must be created, not assumed). A
green lint, type, and test run must never be allowed to stand in for "lexical search works"; nothing in the test
suite touches the database.

**Extension availability is no longer this capability's contract.** The original version of this decision made
extension availability a recurring per-environment precondition owned here, with the branch omitted observably where
an extension was absent. That collided head-on with change 2, which requires a missing capability to be a loud
provisioning failure, and change 2 won — see Coordination point 1. Both the lexical and the fuzzy
precondition requirements were deleted from this change's retrieval capability. What is left here is re-ranking,
index identity, and extraction ordering.

**F8 is CLOSED, and the answer is favourable twice and unfavourable once.** The earlier version of this decision read
the lexical extension question as settled when D14.2 had left one part of it open: whether the access method is
literally named `bm25`. That could not be answered read-only — the probe found no such access method precisely
because the extension was not installed — and it needed the user's authorisation for a `CREATE EXTENSION`. The user
authorised it, scoped to that one statement, and it was run on 2026-08-18 (`findings-database.md` §10). The answers:

- The access method **is** named `bm25`, and it is now present in the catalogue alongside the built-in methods.
- Operator classes are `text_bm25_ops` on `text` (default) and `text_array_bm25_ops` on `text[]` (default); the
  extension supplies the query and vector types; the operator behind `<@>` is a scoring function over
  `(text, bm25query)`.
- `to_bm25query` has **two** overloads: one taking the query text alone, and one taking the query text plus an
  **index name**.
- **The repository's existing lexical SQL is already correct.** It uses the two-argument index-scoped overload, and
  its negation and ordering (`-1 *` on the returned score, `< 0` as the match predicate, ascending order on the raw
  operator) are the right shape for a distance-style operator. This is the third time a plan has declared greenfield
  over working code, and it is the strongest confirmation yet: no lexical SQL is rewritten in this change. **Any task
  that would have rewritten it is a harvest task instead.**
- **Unfavourable, and it is the remaining break:** there is **no `bm25` index anywhere in the database**. Because the
  two-argument overload takes the index name as a *literal argument*, the name is part of the query contract — an
  index of the right shape under a different name does not satisfy the SQL. Creating those indexes by exact name is
  change 0's migration work, so this is Coordination point 5 and a named dependency, not a task this change owns.

The other half of the precondition still closes favourably: the lexical extension is available at version 1.3.0 on
the managed instance the application actually connects to. The vector and vector-index extensions are installed; the
fuzzy, accent-folding, and identifier extensions are available and not installed. The compose database has never been
up in this working copy — nothing listens on its port — so any check aimed at the container image answers a question
nobody asked.

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
schema. **Chosen:** retain the branch and request the extension and index in the foundation change's single migration.

**Revised on the omission question.** The original form of this decision also required the branch to be omitted
observably where its extension or index is absent. That requirement is **deleted** — change 2 owns the missing-
capability contract and it is fail-loudly, not omit-and-continue (Coordination point 1). The consequence for this
change is that the fuzzy branch has no runtime degradation path of its own: either change 0's migration created the
extension and index, or provisioning fails. That is a stricter and more honest contract than the one it replaces, and
it removes the only place where this change told the same code the opposite of what change 2 tells it.

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

**Two boundaries that follow from this, both settled after review.** First, pool ownership: because the lifespan
wiring stays commented, the application is **not** the owner of a checkpointer connection pool, and no requirement
here may imply it is. Ownership belongs to the process that constructs the checkpointer — the worker — and the
capability now says so and additionally requires that the disabled application construction stays disabled
(Coordination point 3). Second, the **consumer-side** fail-closed contract for the shared *checkpointer* is
**change 3's**, not this change's: D17 names the unguarded read of `app.state.langgraph_checkpointer` in the agent
dependency module as the defect and makes it change 3's step 1. This change supplies only the honest provisioning —
setup either returns a usable checkpointer or raises, never an absent value from a function typed to return one
(Coordination point 2). The synchronous *ingestion* surface's fail-closed requirement, which is about a different
shared object on a different router, stays here.

**The one live defect in this area that this change can fix without uncommenting anything** is the shutdown
asymmetry. Teardown is invoked on shutdown while the setup it pairs with is commented out, and teardown's own
early-return on an absent checkpointer is silent — indistinguishable from a successful close. Worse, its pool guard
tests for an attribute that the value it is handed does not have, because the constructor being called returns an
async context manager rather than a saver, so the pool would not be closed even when one existed and nothing would
say so. Making teardown report which of the three outcomes occurred is real work, is import-level and type-level
provable, and touches no commented block.

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

### Decision 14 — The checkpointer is a *consumer* of change 0's accessor set, and there are two URL flavours, not three

Neither existing option works. The raw configured URL carries **no password** — the relational engine works only
because its accessor repairs the URL, injecting the password and stripping the transport parameters its driver
rejects. But that accessor returns the relational engine's **dialect alias**, which the checkpointer driver cannot
parse. One is unauthenticated, the other unparseable.

**Corrected count, replacing the "three flavours" claim this decision originally made.** There are **two**: the
relational engine's dialect-aliased form (SQLAlchemy plus asyncpg), and the plain client-library form (libpq/psycopg),
which must **retain** the transport-security parameter the other driver rejects. The third flavour previously claimed
here — a memory-subsystem form — does not exist: that subsystem is **not a URL consumer at all**. Its configuration
exposes discrete host, port, database, user, and password fields and no connection-string field whatsoever, so a
requirement to hand it a URL is unimplementable against the installed version, not merely inelegant
(`findings-database.md` §9, a retraction issued after reading the installed package). The dialect alias is currently
stripped in two separate places, and a third component reads the raw passwordless URL.

The plain client-library flavour exists **because of this change's checkpointer**. It is the consumer that can parse
neither the raw configured URL nor the dialect-aliased form, so it is the reason that flavour has to exist at all —
worth stating, because the accessor otherwise looks like a foundation-change tidy with no caller.

*Alternatives considered.* (a) Reuse the relational engine's accessor — rejected: unparseable scheme. (b) Read the
configured URL raw — rejected: unauthenticated. (c) Repair the string at each call site — rejected: that is the
current state, and it is why the components disagree.

**Chosen, and narrowed after review:** exactly one accessor per flavour, with credential injection and scheme repair
living only there — and **that accessor set is change 0's**, specified by `infrastructure-client-access`. This
decision originally conceded that the durable fix was change 0's while the spec asserted ownership of it anyway; the
requirement asserting ownership has been **deleted**, and this change is now purely a consumer (Coordination
point 4). What remains local, and is genuinely this change's, is the checkpointer-side contract: it takes its string
from the accessor for its flavour, performs no repair of its own, never receives the dialect alias, and never logs
the string or its credentials.

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

**The archived typed-task-registry capability is harvested, and this records it so the omission is not read as an
oversight.** An archived change shipped a capability binding typed task dispatch: a registry that validates a
dispatched payload against a registered model *at dispatch time*, falling back to a permissive payload for an
unregistered name and logging that fallback. It is **not** present in the live capability directory, so there is no
live requirement block to delta — the same situation as the node-failure-pattern capability in Decision 16, and it is
handled the same way. Its contract is harvested into the worker-deployment capability's task-name requirement rather
than duplicated as a new capability under the same name, which would fork one contract into two at archive time.

Harvesting it also **resolves the review's one residual objection to that capability**: the requirement's original
scenario read "an event dispatches a task name that is not registered", which reaches through the durable-event relay
whose tables this change is forbidden to assume work. The archived registry's dispatch helper is the natural
unit-level seam — invoke the helper directly with an unregistered name, and with a malformed payload for a registered
name — so the check now requires no upstream event at all. The scenario was rewritten accordingly. One deliberate
tightening over the archived text: the archived version let an unregistered name fall through permissively with only
a warning, which is the same invisible-failure shape this change exists to remove, so the requirement now demands the
unregistered name be *reported as a failure* rather than merely logged.

### Decision 19 — Re-ranking is **not** missing: the work is harvest, unify, and fill one gap

The proposal originally called re-ranking "the one genuinely missing third of the hybrid contract", and the disposition
it came from said "add re-ranking (genuinely missing)". Both were wrong, and the correction has been verified directly
against source rather than inferred.

**What actually exists.** A cross-encoder re-ranker class exists, wrapping the sentence-transformer cross-encoding
model with a documented fallback model. It is not an optional extra — it is wired as a **graph edge**: the retrieval
graph adds it as a node and edges `hybrid_postgres → reranker → context_grader`. Its constructor parameter looks
off-by-default (`reranker: CrossEncoderReranker | None = None`), but the node factory resolves
`reranker or CrossEncoderReranker()`, so the node **self-provisions**. Nothing injects one anywhere, and it still runs
on every request through that graph. A **second, independent** call path exists in the documents service, and that one
constructs a fresh instance **per call** — loading a cross-encoder model on every invocation, against the class's own
docstring warning that it is CPU-bound.

*Alternatives considered.* (a) Build a re-ranker as the proposal read — rejected on the facts: it would be the fourth
time this project's planning declared greenfield over working code, and it would produce a second implementation of
something already wired into a graph edge. (b) Leave the two paths alone and only add the missing one — rejected: the
per-call construction is a live performance defect on a shipped surface, and three re-ranking sites is worse than one.

**Chosen: harvest, unify, fill one gap.** The class and the graph node are kept as the foundation. The genuine gap is
exactly one path — the direct hybrid retrieval entry point fuses and hydrates but never re-ranks, while the agentic
entry point goes through the retrieval graph and does. That path is routed through the existing re-ranker. The
per-call construction in the documents service is replaced by the one shared, process-lifetime instance. The
capability therefore requires a single re-ranking implementation, a model loaded once per process, and every ranked
path re-ranking — not the existence of a re-ranker, which is already true.

**Second-order correction, and it narrows a Non-Goal.** Because the re-ranker genuinely needs the sentence-transformer
cross-encoder, the "drop the transformer dependencies" item is not merely unachievable as a whole (Decision 3) — the
sentence-transformer half is **settled: it stays**. Only the tokenizer half of that item remains in scope at all, and
Decision 3's reasoning covers it.

**The standing rule this produced, recorded because it has now cost three corrections** (lexical, rank fusion, and
re-ranking): before any requirement says "add X", grep for X's **edge wiring**, not just its symbol — and follow one
layer past an `| None = None` parameter, because the default is often resolved downstream.

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
checkpointer, a worker-interrogation proof for the queue consumer, unit-level proofs for every correctness fix, and a
checkpoint round-trip against a **local scratch database the task brings up itself**, never against the managed
instance: the checkpointer's setup issues DDL, and every probe this work has made against the managed instance was
read-only. State plainly in the task list that the single upload-to-chunks check is **not** available here, so nobody
writes it as a task and then deletes it when it cannot pass.

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

**[The lexical extension's availability is controlled by the vendor, not by this repository]** → It is available on
the managed instance, and there is **no fallback**: the two alternative lexical extensions are not available at all.
Creating an extension conditionally does not protect against a missing one — it only suppresses "already exists", so a
missing extension aborts the whole migration, taking the durable-event and billing schema with it. **Mitigation, and
it changed after review:** this is now change 2's contract and it is **fail loudly**, not degrade — an absent
extension means change 0's migration did not run, which is a deployment error rather than a runtime condition
(Coordination point 1). The mitigation is therefore ordering and verification, not absorption: change 0 creates the
extensions explicitly, and the residual risk is that its migration aborts wholesale, which is a loud failure with a
named cause. The lexical **index** is the sharper residual: no `bm25` index exists anywhere today, and the index name
is a literal argument inside the query, so change 0's migration must create it under exactly the pinned name
(Coordination point 5).

**[A forward-only contract with no data to test against]** → The requirement that a configured dimension differing
from stored vectors refuses new writes cannot be exercised as a data check: there are zero stored vectors and no
vector columns. **Mitigation:** its verification is a unit test over a stubbed stored width, not a query against a
table. Stated here so the task list does not promise a proof that cannot exist, and so a later reader does not read
the absence of a data check as an omission.

**[A swallowing error handler behind the ingestion service could turn a failed ingestion into a success response]** →
The service logs an exception at one point and it was not established whether it re-raises. **Mitigation:** the task
that touches that file must read the handler, not pattern-match it. An unmounted swallowing handler is invisible; the
same handler behind a live surface is a correctness bug on shipped surface.

**[Scope creep by reading "the retrieval module is in scope" as "collapse the schema now"]** → It is not: the
collapse is the following change and mounting is gated elsewhere. **Mitigation:** the retrieval work here is about
re-ranking, index-name constants, and extraction ordering — **not** about table names, **not** about fusion (change
2 owns the fused contract), and **not** about the lexical SQL, which §10 confirmed is already correct. If a task
starts editing retrieval column definitions, rewriting a fusion rule, or rewriting a lexical query, it has left this
change.

**[Seven new capability directories, and the four-hashtag scenario trap fails silently]** → Three hashtags or bullets
drop the scenario with no error. **Mitigation:** a pattern search for scenario headers with one to three hashtags
across the change's spec files must return zero, run as the final check before validation; it returns zero today.

## Migration Plan

1. **Nothing to back-fill, by accident.** No document, chunk, or search rows exist, and no checkpoint tables exist,
   so the embedding dimension, the chunk shape, the normalisation convention, and the pipeline state shape are all
   settled without data migration. This reasoning is not obvious to a later reader and is recorded here deliberately.
2. **The foundation change lands first**: merge the revision heads, then one new migration creating the target
   schema — the document and chunk tables with the additional timestamp column, the durable-event and dead-letter
   tables, the vector index, and the lexical, fuzzy, accent-folding, and identifier extensions. It must also create
   the lexical **indexes by exact name**, because the index name is a literal argument inside the query text
   (Coordination point 5). Until it does, the lexical branch cannot execute: there is no `bm25` index anywhere.
3. **Install a working client-library binary binding for the checkpointer driver and delete the placeholder alias in
   the same commit.** Without that binding the checkpointer type cannot even be imported, so the alias fallback is the
   live path and is currently the only reason the application boots on this machine. This is a hard precondition for
   any checkpointer work, and nothing else in the change depends on it, so the correctness fixes, the seam
   unifications, and most of the retarget proceed in parallel with it.
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

Numbering is stable, not sequential: question 1 has closed and is recorded below, and the remaining two keep the
numbers they were written with. Five places — C7's task body, two of its Proofs, and two of its evidence bullets —
refer to "Open Question 1" and mean the queue-topology one. Renumbering would silently re-point every one of those
references at a different question.

2. **Does the bare graph builder still accept a validated-model state type on the installed framework version?**
   Inherited and deliberately left open. It does not block this change, which only shrinks channels, but it blocks
   change 3's conversion decision. Resolve from the graph builder's own documentation, not from another pass over
   the repository corpus, which is scoped to the prebuilt agent constructor.
3. **Does the framework validate a fan-out payload against the state schema?** The per-chunk fan-out constructs a
   plain mapping carrying keys the state type forbids as extras, while the graph is compiled against that state
   type. If the framework validates, the fan-out is rejected and the fold hits it immediately. Resolved by the
   checkpoint round-trip check, which invokes the graph once — so it is answered by work already planned, but it is
   recorded because the answer changes the fold's shape.

### Closed since the first draft

- **Question 1 — does ingestion get its own queue, or share the default one?** **CLOSED, answered by the user on
  2026-08-23: a dedicated ingestion queue with its own concurrency, and its own worker service.** This was the one
  question here that no amount of reading could settle — the configuration forbids creating queues implicitly, so the
  queue set is fixed, and an extra queue is an operational commitment with a cost: it needs a consumer, or it
  accumulates silently. It changed C7's content, so C7 was written with the dependency named and held blocked rather
  than defaulting a topology.

  The rationale, which is now recorded in code at every site it constrains: minutes-long model work will otherwise
  starve sub-second billing and transactional-email tasks, and `worker_prefetch_multiplier=1` does **not** prevent
  that — prefetch stops one worker hoarding messages off the broker and does nothing about head-of-line blocking once
  every worker slot is already occupied. Two queues with two disjoint consumer sets is what removes the coupling.

  Four consequences, all landed in C7:
  1. `task_queues` gained a third quorum queue on the existing task exchange with its own routing key, dead-lettering
     to the **existing** dead-letter exchange rather than a fourth queue nobody watches.
  2. `task_routes` is derived from the declared task list, with the three ingest names taken from a single
     `INGESTION_TASK_NAMES` set. The `tasks.*` glob is gone, so all 16 names are now routed explicitly — which the
     capability required and nothing else in this change would have delivered.
  3. The deployment is **three** services: a default-queue worker at concurrency 8, an ingestion worker at
     concurrency 2, and the scheduler. `-Q` on both workers is mandatory, because a worker without it consumes every
     declared queue including the dead-letter queue.
  4. The capability's "long work must not starve latency-sensitive work" requirement is now provable without a broker:
     disjoint queues with disjoint consumers is a structural guarantee, not a measurement.

  `LEGAL_BATCH_EXTRACTION` is also minutes-long model work and was **not** moved: the answer named the three ingest
  names, and a further queue is a further decision to be asked for. It is listed here rather than deleted so that a
  reader who remembers this as open can see how it closed.

- **F8 — is the lexical extension's index access method literally named `bm25`?** **CLOSED, answered `bm25`.** It was
  the one question here that could not be settled read-only; it needed the user's authorisation for a single
  `CREATE EXTENSION` against the live database. The authorisation was given and scoped to that one statement, and the
  probe ran on 2026-08-18. The answer, the operator classes, the two `to_bm25query` overloads, and the two
  consequences — that the repository's lexical SQL is already correct, and that no `bm25` index exists anywhere — are
  recorded in Decision 6 and Coordination point 5. It is listed here rather than deleted so that a reader who
  remembers it as open can see how it closed.

