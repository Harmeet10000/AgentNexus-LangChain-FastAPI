# Architecture Decision Records — cleanup-foundation

Six decisions in this change outlive it. They are recorded here rather than in `design.md` because they constrain
work that has nothing to do with change 0: the shape of the migration history for as long as this database lives, how
every future endpoint learns who is calling it, how every future consumer of the database connection string obtains
one, why an index name cannot be changed freely, what the event relay owes when one of its relations is absent, and
by which of two mutually exclusive routes a relation whose creating revision is falsely recorded as applied is ever
brought into existence. `design.md` records what change 0 does; these record what remains true afterwards.

---

## ADR-1: Repair the migration chain by joining heads plus one authoritative revision, and accept the chain as permanently misleading

**Status:** Accepted

### Context

The Alembic history and the live database have diverged, and not marginally. The live Timescale Cloud instance is
stamped at `0004`, and three revisions in the chain are recorded as applied while having created nothing at all:
`0001_add_outbox_tables`, `8a7d9b1c2e3f`, and `9f4a1b7c6d2e`. The entire document, chunk, vector and search schema
that those revisions describe was never created. Neither were the event-outbox relations, which is why
`POST /auth/forgot-password` and `POST /auth/resend-verification` return `500` today — after having already persisted
a token, so the failure is not even clean.

The chain additionally exposes more than one head, so any command that needs an unambiguous target — `upgrade head`,
an offline render of it, a schema comparison — fails on the ambiguity before it can do anything useful. Anything that
reads the history as a description of the database is reading fiction.

One thing the divergence does **not** break, and it is worth stating because a plausible-sounding claim to the
contrary was recorded here earlier and has been withdrawn: the history *can* be rendered offline from base. An offline
render emits DDL as text and never executes it, so a revision that ALTERs a relation nothing created renders its
ALTER perfectly happily. Offline rendering is therefore available as a proof mechanism, with one limitation that
matters more than the imagined one: because it has no database from which to read a recorded version, it always
starts at base and always emits every revision. See ADR-4's sibling constraint in `design.md` D14.3.

Fifteen billing tables, by contrast, genuinely exist and genuinely hold structure that matters.

### Decision

Repair forward. Join the divergent heads with a merge revision, and add **one** authoritative revision that creates
the target schema — outbox relations first, then the document schema — conditionally throughout, so it succeeds
whether a relation is absent or already present. The three hollow revisions stay in the chain, stay stamped, and stay
untouched.

The chain is thereby accepted as **permanently misleading**: it will forever contain revisions that claim to have
created relations they did not create.

### Alternatives considered

**Edit the unapplied revisions in place.** Rejected by the user. The revisions are recorded as applied. There is no
way to prove that no other environment — a developer machine, a CI database, a staging instance, a copy taken before
the divergence — actually applied them. Editing a revision that some other database has already run produces silent,
undetectable schema drift in that database, and the blast radius is unbounded precisely because it is unknowable.

**Full rebaseline / squash of the history.** Rejected by the user. The fifteen billing tables that genuinely exist
would have to be hand-reconciled against a rewritten root revision, table by table, column by column, constraint by
constraint. That is a large, delicate, entirely manual verification exercise whose failure mode is a production
schema that no longer matches its own declared baseline — a strictly worse position than a misleading history.

### Consequences

- **`downgrade` past the joined head is forbidden.** Not discouraged — forbidden. The reversals of the three stamped
  revisions would drop relations that were never created, and the reversal of the authoritative revision would drop
  relations those revisions claim to own. State this in the runbook without softening it.
- **The revision history no longer describes the database.** Reading the chain to learn the schema will mislead. The
  database's own catalog is the only authority. Any future tooling that derives expectations from the migration
  history — autogenerate diffs included — must be treated as unreliable on the affected relations.
- **The revision that will actually execute is not the one the chain makes prominent.** One branch head is unstamped
  and therefore runs on the next upgrade, ahead of both the merge and the authoritative revision. Reasoning about
  "what the upgrade does" must start from that revision, not from the authoritative one, and a failure there aborts
  the upgrade before the authoritative revision's outbox repair is reached.
- A fresh environment cannot simply run `upgrade head` and trust the result; it needs the documented procedure that
  names the revisions it deliberately skips and the extensions it must create beforehand.
- The squash remains available later, from a healthier starting point: once one authoritative revision describes the
  real schema, a future rebaseline has something truthful to baseline against.

---

## ADR-2: Derive request identity from validated token claims, not from mutable request state

**Status:** Accepted

### Context

Endpoints have been reading the caller's identity out of request-scoped mutable state populated earlier in the
request lifecycle, rather than from the claims of the token that was actually validated. The two are not the same
thing. Mutable state can be set by any earlier participant in the chain, can be absent when the ordering changes, and
carries no cryptographic relationship to the credential the caller presented. Authorization decisions made against it
are decisions made against whatever the last writer put there.

In this repository the state in question is written by nothing at all, so every read of it raises on the first
request.

### Decision

Request identity is derived from the **validated token's claims**. The authenticated principal reaches a handler
through the dependency that verified the token, and nothing else is a source of identity. Request-scoped mutable
state may carry correlation and diagnostic data; it does not carry who the caller is.

### Alternatives considered

**Keep reading mutable request state, and add ordering guarantees so it is always populated first.** Rejected: this
makes correctness depend on middleware ordering, which is invisible at every call site that consumes the value. The
defect returns the first time ordering changes, and it returns silently.

**Validate the token again at each consuming site.** Rejected as redundant work with a worse failure mode — two
validation paths that can disagree, and a per-request cost paid repeatedly for a value already computed.

### Consequences

- **BREAKING, and observable by callers: `500` becomes `401`.** Six mounted document endpoints currently answer
  `500 Internal Server Error` to an unauthenticated request, because reading the unset state raises before any
  authentication decision is made. After this decision they answer `401 Unauthorized`. This is the correct response
  and it is the point of the change, but it is a behaviour change on shipped surface: a client that treats `500` as
  retryable, with backoff, now receives a terminal failure it must stop retrying. Anyone measuring endpoint health by
  5xx rate will see that rate drop and the 4xx rate rise for the same traffic. Record it as a breaking change rather
  than as a bug fix, because from the caller's side those are indistinguishable except by the changelog.
- This closes a **class** of defect, not an instance. Any future endpoint that needs the caller's identity has one
  legitimate source, so the mistake is no longer available to make.
- Handlers that need identity must declare the authentication dependency, which makes the requirement explicit in the
  signature instead of implicit in the middleware stack — a readability gain and a testability gain.
- Anything that legitimately needs to run before authentication cannot consult identity. That is the correct
  constraint, and any code that appears to need otherwise is misplaced.

---

## ADR-3: One connection accessor, two URL flavours, and discrete fields for consumers that take no URL

**Status:** Accepted. Supersedes an earlier draft of this ADR that claimed three URL flavours; that count was wrong
and the premise behind it has been retracted at source.

### Context

Consumers of the database connection need it in more than one form, and a single string cannot serve them all.
Enumerated against the installed packages rather than assumed, there are exactly **two** URL forms:

1. **SQLAlchemy with the asynchronous driver** — the form the application's own connection pool needs, with the
   driver qualifier in the scheme and with the connection parameters that driver rejects as query arguments removed.
2. **Plain libpq** — the form a low-level driver accepts directly, with no driver qualifier in the scheme, retaining
   the transport-security parameter that driver does want.

A third flavour was recorded here earlier, for an embedded third-party memory component. **It does not exist.** That
component's configuration object exposes only discrete fields — host, port, user, credential, database name, provider
— and carries no connection-string setting of any kind, so there is nothing on it that a URL could be assigned to.
The call site already passes the discrete fields. The "third flavour" was inferred from a second, similarly-named
local dictionary in the same function that holds a URL and is only ever returned, never consumed as configuration.
Building the third flavour would have produced API surface with no possible caller, and — worse for an ADR meant to
bind future work — it would have sent the next person adding a consumer looking for a URL when what they need is
fields.

The current code demonstrates what a single string does under this pressure. The lifespan wiring strips the driver
qualifier from the URL before handing it onward, and the relay strips it **again** from the already-stripped value —
a defensive second strip that is a no-op precisely because the first one happened, and that exists only because
neither site can tell what form it was handed. Every consumer performs string surgery on a value whose shape it
cannot verify, and the surgery is silently idempotent, which is why the duplication went unnoticed.

### Decision

One accessor owns the database connection configuration and exposes:

- the **two** URL flavours above, selected explicitly by name, and
- the same underlying values as **discrete fields**, for consumers that assemble their own connection.

Consumers request what they need by name. No consumer performs string manipulation on a connection URL, and no
consumer is handed a URL it cannot use.

### Alternatives considered

**One canonical string plus a documented normalisation helper.** Rejected: a helper still leaves each call site
deciding whether to call it, which is the current failure re-expressed. The double-strip proves that call sites
cannot reliably determine the shape of what they were handed.

**One accessor per consumer, independently constructed from settings.** Rejected: independent constructions of the
same value drift. A credential rotation or host change then has several places to land, and the last one is
discovered in production. This is not hypothetical here: the embedded component's call site already reads the host
and database name from settings independently of the accessor, so it can be pointed at a *different database than the
application* with a valid credential and succeed silently. Serving its discrete fields from the same accessor is what
closes that.

**Keep the third URL flavour anyway, as future-proofing.** Rejected: an unused flavour is untested surface that
documents a false expectation about what its consumer accepts. If a future consumer genuinely needs a third form, add
it then, with that consumer as its test.

### Consequences

- String manipulation of connection URLs becomes a reviewable defect with a named alternative; the double-strip and
  every future instance of it disappear.
- The flavour set is **closed at two** by decision, not left open. Adding a third requires a named consumer that
  accepts a URL, which is the check that would have caught this ADR's original error.
- Credentials, host, and options are configured once and are guaranteed consistent across both URL forms *and* the
  discrete fields — which the current arrangement does not guarantee, it merely happens to achieve for the URL forms
  and does not achieve at all for the fields.
- The accessor becomes a chokepoint worth its own tests, since a defect in it is a defect in every database
  consumer simultaneously.
- **Method note worth keeping:** the three-flavour count survived several planning documents because it was stated as
  a count rather than as a list of named consumers. Counts do not carry their own evidence; a list of consumers with
  the setting each one accepts does. Prefer the list.

---

## ADR-4: A retrieval index name is part of the query contract, not a migration-local convention

**Status:** Accepted

### Context

Almost every index in a schema is freely renameable: the planner selects it by shape, and no application code names
it. That assumption is correct here for the vector and trigram indexes, and **wrong for keyword ranking**.

The keyword-ranking extension this repository uses exposes a two-argument form of its query constructor whose second
argument is the **index name, as a literal string in the SQL text**. The repository uses that form in nine places, and
one of the names is additionally pinned in a module-level constant. The query is well-formed and the operator is
real; the extension, its access method and its operator classes are all present and the existing SQL is correct
against the installed version. Yet keyword retrieval returns nothing, because no index of that access method exists
anywhere in the database under any name.

The failure mode is what makes this ADR necessary. An index of exactly the right access method, on exactly the right
column, with exactly the right operator class, created under a different name, does not raise an error. It returns
zero rows. There is no exception, no log line, and no plan difference that a shape-based check would notice — the
query simply becomes a very efficient way of finding nothing. A schema check that verifies "a keyword index exists on
this column" passes while retrieval is entirely broken.

### Decision

For any index whose name appears in query text, **the name is part of the contract between the migration and the
query**, and is treated with the same care as a column name:

- The migration creates it under the exact name the query text names, and the change record states the name.
- Renaming it is a change to the query contract. Every query literal and constant that names it changes in the same
  commit, or the rename does not happen.
- A schema check for such an index asserts the **name**, not only the shape.
- Where a query literal names an index on a relation a change deliberately does not create, that change does not
  create the index under a substitute name and does not edit the literal either; it names the change that owns the
  retarget. Splitting the literal and the index between two changes is how a half-migration becomes permanent.

### Alternatives considered

**Rename the indexes to something tidier and update the constants.** Rejected for this change, not in principle: the
retarget of those literals onto the unified relations is owned by a later, sequenced change, and doing half of it
here would leave the other half orphaned. Also rejected as a default habit — every rename costs a coordinated
multi-file commit, so the names are worth keeping stable rather than tidy.

**Use the single-argument form of the query constructor so no index name appears in SQL.** Rejected as out of scope
and as a larger behavioural change than it looks: the index-scoped form is what makes the ranking deterministic
against a chosen index, and switching forms changes retrieval behaviour, not just syntax. It remains available as a
future option, and if taken it *dissolves* this ADR — which is the cleanest possible resolution and is recorded here
so a future author sees it.

**Assert index shape only, and treat names as free.** Rejected: that is the assumption that produced the current
silent failure.

### Consequences

- Index naming for retrieval indexes joins the small set of cross-layer string contracts in this repository, alongside
  route paths and event-type names. It is reviewable as such.
- The migration and the query text are coupled, and the coupling is now written down. Any future author who renames a
  retrieval index for tidiness will find the constraint before shipping rather than after retrieval quietly stops
  returning rows.
- Maintenance functions in the same extension family are **also** index-name keyed — a force-merge call in the
  ingestion path names one by literal string — so this contract extends beyond retrieval to maintenance tasks.
- A missing index and a misnamed index are the same observable event: zero rows. Any diagnosis of "keyword search
  returns nothing" must check the name first, because everything else will look correct.

---

## ADR-5: What the event relay owes when one of its relations is absent — and which spec wins until it owes it

**Status:** Accepted

### Context

The event relay reads and writes relations that do not exist in the deployed database. It does not fail. A broad
`except` in its startup scan absorbs the undefined-relation error, and the long-running notification listener runs
inside a detached background task where nothing observes its outcome. The application therefore boots successfully,
reports itself healthy, and has a **permanently dead outbox** for the lifetime of the process: nothing retries,
nothing is published, and the only trace is two warning lines.

Meanwhile two mounted public endpoints and the document upload path enqueue events *without* that protection. They
persist their own state change and then fail on the insert, returning `500` after a partial write — a reset token
that no email will deliver.

So the same missing relation produces silent permanent degradation on one path and a loud partial write on another.
That asymmetry is not designed; it is an accident of which code happens to sit inside a broad handler.

There is a further complication, and it is the reason this ADR exists rather than a requirement. An accepted
capability in this repository **explicitly sanctions** the broad catch at outbox-relay degradation boundaries: a
requirement stating that degradation boundaries keep `except Exception` with an added note, carrying a scenario about
the relay dead-lettering on any failure. A requirement demanding loud failure and a requirement sanctioning silent
absorption cannot both be in force over the same lines. An earlier draft of change 0 added the loud-failure
requirement anyway, in a different capability, with no step implementing it — which would have archived two accepted
specs that disagree, and left an implementer with no way to know which one to satisfy.

### Decision

Three parts, in force together.

**1. The target state.** When a relation the relay depends on is absent, the relay owes: a report at error severity,
distinguishable from a transient connection failure; the missing relation named; the outbox subsystem recorded as
unavailable in a form the readiness surface can observe; and the same treatment for a detached listener that
terminates. Absence of a schema object is a defect, not a degradation, and the handler that cannot tell those apart is
too wide.

**2. The precedence rule, until the target state lands.** The sanctioning requirement **wins**. The broad catch stays,
and no change may assert loud failure as a property of the current system. This is not a preference for silence; it is
a refusal to hold two contradictory specs. The gap is carried as an explicit Non-Goal in the change that discovered
it, so it is a recorded debt rather than an unwritten one.

**3. What the change that narrows the handler must ship, all of it, in one change.** Narrowing the handler is not a
code-only edit:

- The revision that creates the relations must already be applied in every environment the narrowing reaches.
  Narrowing first converts a silent permanent degradation into a **boot failure**, because the wrapper around relay
  startup does not catch database errors.
- It must ship a `MODIFIED` delta retiring the scenario that sanctions the broad catch at this boundary — reproducing
  that requirement's every remaining scenario title verbatim, since a `MODIFIED` block replaces the requirement
  wholesale.
- It must ship the loud-failure requirement and the code that satisfies it **in the same change**, so the requirement
  is never in force without an implementation.
- It should decide, explicitly, whether the endpoint-side property belongs with it — that a state change must not
  remain committed without its event. That is a transaction-boundary change in the auth service, not a relay change,
  and it may reasonably be a separate change; what it may not be is assumed.

### Alternatives considered

**Add the loud-failure requirement to change 0 and leave the contradiction for later.** Rejected. Archiving a
requirement with no implementing step, contradicting an accepted spec, is the same defect as archiving a false
one — the deployed spec of record becomes something no code satisfies, and the next change reads it as truth.

**Narrow the handler inside change 0, ordered after the revision.** Rejected as scope, not as engineering. It is
defensible and it would work, but it pulls exception-tightening, a readiness-surface addition, and a paired spec
retirement into a change whose job is schema, identity and subtraction. The ordering constraint that makes it safe is
recorded here, so the change that does it inherits the reasoning rather than rediscovering it.

**Drop the loud-failure ambition and accept the broad catch permanently.** Rejected: the current behaviour makes a
schema defect indistinguishable from a healthy relay on the readiness surface, and the same handler breadth will hide
the next schema drift identically.

### Consequences

- An implementer reading change 0 has an unambiguous answer to "which spec wins": the sanctioning one, until a named
  later change replaces it. No one has to guess.
- The debt is bounded and specific: one handler, one listener, one scenario retirement, one requirement, one ordering
  precondition. It is small enough to be done deliberately and too easy to do accidentally in the wrong order, which
  is why the order is written down.
- Until it lands, the outbox's health is **not** observable from the readiness surface. Anyone diagnosing "events are
  not being delivered" must check the relation's existence directly; a green health report is not evidence.
- The asymmetry stays visible: the same missing relation yields `500` on the enqueue paths and silence on the relay
  path. That is the clearest available illustration of why "the application boots" is not a correctness signal, and it
  is worth keeping in the record even after the repair.

---

## ADR-6: A relation whose creating revision is falsely recorded as applied is created by a forward repair revision, never by rewinding the version pointer

**Status:** Accepted

### Context

The deployed database was **stamped**, not migrated. Measured read-only: `alembic_version` holds exactly one row,
`0004`; the public schema holds **16** tables — the fifteen billing tables `0002` creates, plus `alembic_version`
itself. Every one of the eleven relations the sequenced work depends on is **absent**: `document_vectors`,
`chat_messages`, `chat_sessions`, `search_documents`, `search_chunks`, `parent_documents`, `clauses`,
`outbox_events`, `dead_letter_events`, `documents`, `chunks`. There are zero `bm25` indexes.

Two structural facts turn that from an inconvenience into a decision that outlives this change.

**First, the stamp was a workaround for a broken migration, not a deployment shortcut.** `9f4a1b7c6d2e` cannot run
against any database. Its `op.create_table` creates **`parent_documents`**; but the rest of its body operates
throughout on **`clauses`** — a `batch_alter_table`, two `UPDATE`s, three `alter_column`s, a foreign key, four
indexes, a `bm25` index and a `diskann` index. **No revision in the chain creates `clauses`, and no ORM model
declares it**; `clauses` appears in exactly one file in the entire versions directory, the revision that mutates it.
An upgrade that reaches `9f4a1b7c6d2e` therefore dies with `UndefinedTable`. The independent proof that it never
executed is that it is marked applied while `parent_documents` — the table it *does* create — is absent.

**Second, the chain is branched, so the version pointer's own ancestry decides what is reachable.** `2bc7726317f6`
has two children: `8a7d9b1c2e3f` (leading through `9f4a1b7c6d2e` and `0001`–`0004`) and `a71f0d7d9c12`. The stamped
pointer sits on the first branch. So `a71f0d7d9c12` is unapplied, its `down_revision` is satisfied, and it *will*
run — it is the **one** revision a real upgrade executes today, and it creates `documents` and `chunks` and nothing
else. Every other absent relation belongs to a revision the pointer already claims, which `upgrade` will therefore
skip **forever**. Absence plus "already applied" is a permanent state under forward migration; it does not decay.

Exactly two routes out exist, and they are mutually exclusive in practice.

### Decision

**Create the missing relations with a forward repair revision, written idempotently. Do not rewind the version
pointer below `8a7d9b1c2e3f`.**

Concretely, and binding on every later change in the sequence:

1. **The authoritative revision creates, idempotently, every relation a live path needs that a falsely-applied
   revision was supposed to create** — the event-outbox pair above all, since two mounted public endpoints write to
   it. Idempotence is not politeness here: it is what makes the revision safe on an instance where `a71f0d7d9c12`
   already created `documents` and `chunks`, and on a fresh instance where nothing exists. The same revision must
   converge from both states.
2. **The stamp-down route is rejected, and rejected for a specific reason rather than on taste.** Rewinding below
   `8a7d9b1c2e3f` and upgrading for real re-enters `9f4a1b7c6d2e`, which is unrunnable. The route is not merely
   slower; it does not terminate. It stays unavailable until `clauses` is resolved, and resolving `clauses` is not a
   migration question at all — see point 4.
3. **A relation nothing reads is not repaired.** The repair set is the relations on live paths, as *live* is defined
   normatively in this change's `migration-chain-integrity` delta. `search_documents`, `search_chunks`,
   `parent_documents` and `clauses` are deliberately **not** created: their only readers are code a later change
   retargets. Creating them would manufacture the mirror defect — a relation with no surviving reader — while the
   change that would have read them is already scheduled to stop.
4. **`clauses` is resolved by retargeting its readers, not by creating it.** `dispositions.md` item 184 records
   Option **A+**: point the clause readers at the unified store. That decision is now load-bearing for the migration
   chain, because it is what would eventually make the stamp-down route runnable, and it is owned by the
   search-consolidation change together with the `clauses_bm25_idx` literal its four readers name. This change
   creates no `clauses` DDL and edits none of those literals.
5. **The forward-repair route accepts a permanent cost, stated once:** the chain will contain revisions recorded as
   applied that created nothing, and a later revision that creates what they claimed. Reversal below the joined head
   is therefore unsupported for the life of this database, because those reversals would drop relations they never
   created. The only clean exit is squashing the history into one honest baseline, which the user rejected for this
   change and which is recorded as a post-refactor candidate.

### Alternatives considered

- **`alembic stamp base` then a real `upgrade`.** Rejected twice over: it re-enters the unrunnable
  `9f4a1b7c6d2e`, and before even reaching it, `0002` would attempt to create the fifteen billing tables that
  genuinely exist, failing with `DuplicateTableError` on the first of them. The route fails at both ends.
- **Stamp down to `2bc7726317f6` only** — below the fork but above the billing tables. Rejected: it still re-enters
  `9f4a1b7c6d2e` on the way up, and it additionally discards the pointer's record of `0002`–`0004` having genuinely
  been applied, converting an honest part of the history into a second lie to keep the dishonest part consistent.
- **Edit `9f4a1b7c6d2e` to create `clauses` before mutating it.** Rejected: editing an applied revision is forbidden
  by this change's constraints, and the edit would manufacture a relation whose readers a later change is already
  scheduled to retarget away. It fixes a rendering problem by creating a schema problem.
- **Delete `9f4a1b7c6d2e` from the versions directory.** Rejected: the deployed pointer's ancestry names it, so its
  absence makes the deployed `alembic_version` unresolvable — the tool would report the current revision as not
  present in the chain, which is strictly worse than an unrunnable revision that nothing will run.
- **Repair by hand-written SQL applied outside Alembic.** Rejected: it leaves the version pointer and the schema
  disagreeing in a *new* way, which is the precise defect this ADR exists to stop repeating. A repair that is not a
  revision cannot be replayed on the next environment.

### Consequences

- **`upgrade` is never the whole answer again on this database.** Anyone who asks "will migrating produce the schema?"
  must be answered with the version pointer's ancestry in hand, not with the revision graph alone. The two differ, and
  the difference is the whole problem.
- **`alembic upgrade head` — singular — is ambiguous while two heads exist and exits 255 with *Multiple head revisions
  are present*.** Three committed call sites use the singular form: `Makefile:39`, `README.md:272`, and
  `.github/workflows/test.yml:105`. The merge revision makes the singular form resolve again, so those call sites are
  fixed by the merge rather than by editing them — but a chain that forks again silently re-breaks all three, which is
  a reason to treat the single-head property as a checked invariant and not a one-time repair.
- **Only `heads` is well-defined in the meantime**, including for offline renders. Measured: `upgrade head --sql` →
  exit 255; `upgrade heads --sql` → exit 0, 697 lines.
- **The repair revision is load-bearing for a public endpoint**, so its ordering is internal to itself: the
  event-outbox relations are created before the document schema, so a failure in the larger half cannot withhold the
  repair of the endpoints that are returning `500` today.
- **The one revision that will actually execute is also the one carrying the extension hazard.** `a71f0d7d9c12` builds
  a `diskann` index and creates no extension; it is ordered *ahead* of the merge and the repair, so the repair cannot
  fix it, and a failure there aborts before the outbox repair runs. The extension's presence is a precondition of the
  upgrade, asserted before it, on every instance.
- **Applying any of this to the live database requires an authorization this change does not have.** The repair is
  written, rendered and rehearsed against a scratch database under change 0; the deployed upgrade is a separate,
  explicitly authorized act. Nothing in the sequence assumes it has already happened.
