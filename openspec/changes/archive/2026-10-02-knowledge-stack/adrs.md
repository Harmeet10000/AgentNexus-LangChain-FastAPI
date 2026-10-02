# ADRs — knowledge-stack

## ADR-001 — Replace the external structural service; do not wire the client that already exists

**Status:** accepted (user ruling: PageIndex is an idea, not a dependency).

**Context.** `src/app/shared/rag/pageindex/` is a complete client for an external
reasoning-over-document-tree service, re-exported at `shared/rag/__init__.py:3`, with its construction
sitting commented out at `lifespan.py:538`. It is one uncommented line away from working. Meanwhile the
document parser already produces a structural tree for every document during ingestion.

**Decision.** Do not wire the client. Build a local navigator over the tree this system holds, and
delete the client, its re-export, and its construction site.

**Consequences.** No external service enters the retrieval path, and **no document content leaves the
system** to a third party — which is the decisive consideration for a legal corpus, not the per-query
cost. A second copy of each document's structure is not held elsewhere. The cost is that the navigator
must be written rather than called, and that whatever tree-reasoning sophistication the external service
has is not obtained. The rejected shape was recorded explicitly during planning — *uncomment
`lifespan.py:538`, wire the existing client, defer the tree work* — and it is rejected precisely because
it adds the dependency this decision removes.

## ADR-002 — Extraction runs before chunking

**Status:** accepted.

**Context.** Structured extraction could run on the raw parsed document or on the chunks. Both are
implementable.

**Decision.** Extraction runs **before** chunking, and its output is available to the stage that writes
chunks.

**Consequences.** Both stages derive from a single parse, so an extracted entity and the chunk
describing it are associated while one stage still holds both. The alternative leaves entities located
in one coordinate system and chunks in another, requiring reconciliation code between two views of one
document — the kind of mapping that passes on fixtures and drifts on a real corpus. The cost is that the
ingestion path gains a stage before its existing first substantive stage, which is why task 1.3 records
the stage boundary explicitly instead of inserting into the middle of an existing function.

## ADR-003 — Whether the tree is persisted is measured before the second half is designed

**Status:** accepted.

**Context.** The parser produces a structural tree during ingestion. Whether that tree survives into
storage is a separate question, and nothing in the planning established it. Planning noted the risk in
so many words: task 1.2 may reveal the tree is never persisted, converting this change from wiring into
a storage build.

**Decision.** Task 1.2 measures it, first, before any tree-reasoning work is designed. The proposal
states both branches rather than picking the likely one.

**Consequences.** A *no* makes task group 2 mandatory — a storage column plus a migration whose
`down_revision` must come from `ingestion-chunking`'s head. A *yes* makes group 2 a no-op and group 4 a
navigator over data that already exists. Either way the work is scoped before it is started, rather
than discovering a migration halfway through a change sized as wiring. The cost is that this change's
size is genuinely unknown until its first substantive task completes, which is stated in the proposal
rather than hidden. Task 2.3 requires recording which branch was taken, so an unticked group 2 is never
ambiguous between *not needed* and *not done*.

## ADR-004 — Extraction failure is a typed value, is non-fatal, and is distinguishable from an empty result

**Status:** accepted.

**Context.** Extraction depends on an external provider. The natural implementation has two states —
entities, or no entities — which silently merges *the provider was down* with *this document contained
nothing to extract*.

**Decision.** Three outcomes. Success with entities; success with none; failure. Failure is represented
as a typed value the ingestion path handles explicitly, ingestion still completes, and only the failure
case sets the extraction-incomplete flag.

**Consequences.** A document ingested during a provider outage is identifiable and re-runnable. Merging
the two cases would be silent **and permanent** — nothing later would prompt a re-run, because nothing
would look wrong. Representing failure as a value rather than a caught exception is what makes the third
state expressible at all; a `try`/`except` that logs and continues collapses back to two states. The
cost is a flag on the document and two tests instead of one (task 5.2), the second of which asserts a
*negative*: an empty success leaves the flag unset.

## ADR-005 — Canonicalisation is satisfied, not re-specified

**Status:** accepted.

**Context.** `graph-entity-canonicalisation` already requires that canonical graph writes are
idempotent and that re-ingestion does not duplicate entities. Graphiti is built and wired, and it has a
writer.

**Decision.** Route extracted episodes through that existing writer. Write **no** canonicalisation or
deduplication requirement in this change's capability.

**Consequences.** One rule about idempotent graph writes exists in one capability, so the two cannot
drift when either is edited. This change's re-ingestion scenario is a statement of what it must satisfy,
not a second definition of how. Task 5.4's Proof enforces it mechanically by grepping this change's own
spec directory for canonicalisation vocabulary. It is the same discipline `ingestion-chunking` applies
to the embedder: **restore, cite, and repair rather than re-specify.**

## ADR-006 — The navigator is pure, and it is reached through the repository

**Status:** accepted.

**Context.** Tree navigation could be written as a repository method that queries and walks in one step,
or as a pure function over an already-loaded tree.

**Decision.** A pure function from a tree and a query to node paths, performing no I/O, exposed as a
retrieval branch through the repository and the service layer — never from a route handler.

**Consequences.** The navigation logic is testable with a fixture tree and no database, which is what
makes the "correct section path" assertion exact rather than incidental. It also keeps the change inside
the project's layering rule, and task 4.2's Proof checks the router for a repository import to catch the
shortcut. The cost is one extra function boundary between the query and the walk — which is also what
makes the walk reviewable.

## ADR-007 — The external package is deleted, not deprecated

**Status:** accepted.

**Context.** The retiring client could be left importable-but-unused, marked deprecated, or removed.

**Decision.** Remove the re-export, the commented construction, and the package.

**Consequences.** No one wires it back on later believing it is a supported path — which is exactly what
its current state invites, since a commented construction line reads as *nearly finished* rather than
*decided against*. Deletion also makes the decision visible in the diff. The cost is that recovering it
means reading git history; accepted, because the reason it is going is a decision about dependencies,
not a judgement that the code is bad.

## ADR-008 — The commented construction crosses a change boundary deliberately

**Status:** accepted.

**Context.** `lifespan.py:538` falls inside `graph-lifecycle`'s territory — it is that change's file and
that change's concern. Yet the decision to retire the client belongs here.

**Decision.** `graph-lifecycle` leaves the line untouched and asserts it is unchanged. This change
removes it.

**Consequences.** A single commented line gets a two-change handshake, which is more ceremony than one
line warrants. The alternative is worse: `graph-lifecycle` provisioning a client that this change then
deletes — churn, plus a window during which document content is being sent to a third party for a
capability that is about to be built locally. The cost is that a reader of `graph-lifecycle` sees a task
that deliberately does nothing, which is why its Proof states which change is coming for the line.
