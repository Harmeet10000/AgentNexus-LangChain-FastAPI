> Change class: **L** cross-cutting (multi-module, migration, security boundary, public API)

## Why

The repository has accumulated a register of invisible failures: the migration chain has two heads and a
deployed database that was *stamped* rather than migrated, so several relations live code reads and writes do
not exist; and three mounted endpoint groups raise `AttributeError` on their first request because they read
application state under names the startup sequence never assigns. Alongside that sit roughly 2,900 lines of
provably unreachable code whose presence makes every later change harder to reason about. This change is the
foundation the four sequenced changes that follow are built on: it makes the schema deployable, makes the
already-shipped surface answer correctly, and subtracts the dead weight.

## What Changes

- **Migration chain repair.** Join the two migration heads into one, then add a single revision that is
  authoritative for the target relational schema: the unified `documents` / `chunks` relations, the
  `chunks.updated_at` column those relations lack, the vector / keyword / fuzzy retrieval indexes, and the
  event-outbox relations that live write paths already depend on. The revision is written idempotently so it
  converges whether or not an earlier revision already created part of the schema, and it destroys no rows.
- **Model registration.** Every persisted model that live code uses is registered on the single metadata the
  schema-comparison tooling reads, so a future comparison can never propose dropping a live relation. A private,
  orphaned registry declaring six models is retired by **deletion, not harvest**: none of the six has an importer
  anywhere in the application, and moving them onto the shared registry would schedule creation of relations nothing
  reads — the mirror image of the defect this change exists to close. The migration environment's unreachable import
  fallback is deleted so a broken registration fails loudly.
- **BREAKING — request identity.** The caller's user id is derived from validated access-token claims instead
  of from request state that nothing assigns. Six mounted document endpoints change failure mode from
  `500 Internal Server Error` to `401 Unauthorized` for unauthenticated calls. Clients treating the 500 as
  retryable will now see a terminal 401. No request or response schema changes.
- **Startup client access.** Profile endpoints read their object-store and document-store clients under the
  names the startup sequence actually publishes, and answer `503 Service Unavailable` when an optional client
  failed to initialise instead of raising an attribute error.
- **Usable database URLs.** Every database consumer obtains its connection URL from a single accessor, in the
  flavour its driver requires. Two consumers currently read the raw configured value, which carries no
  credentials; a third derives a driver-specific variant by ad-hoc string replacement. There are exactly **two**
  URL flavours — the async-ORM form and the plain low-level-driver form — and the accessor additionally exposes the
  same underlying values as **discrete fields** for an embedded component that accepts no connection URL at all.
- **BREAKING (additive) — health reporting.** The versioned health endpoint reports the graph-memory
  dependency the startup sequence degrades past silently. An absent optional dependency reports
  `not_configured` and does not change the overall status or HTTP status code. Two response shapes (v1 and v2)
  gain fields; nothing is renamed or removed.
- **Subtraction.** **Seven** proven-dead trees and files are deleted, and the list is exhaustive: (1) an
  unparseable 783-line draft, (2) an inverted 36-line parser, (3) a zero-byte vector-store package, (4) a zero-byte
  orchestration-type package, (5) a zero-byte `knowledge_base` feature package, (6) a zero-byte `web_scraping`
  feature package, and (7) the 1,129-line reconciliation subsystem including the private-registry schema module that
  sits inside it. Four of the seven carry a coupled `__init__` or config edit, made in the same commit as the
  deletion, so no commit leaves the application unbootable.
- **Annotation residue.** Two `object`-typed parameters inside the blast radius are given real types.

## Scope / Non-Goals

In scope: the migration chain and the target schema DDL, model registration, request identity, startup client
access, database URL access, health reporting, and the deletion set with its coupled edits.

Out of scope, deliberately: promoting the ingestion pipeline (change 1); consolidating the search relations
onto the unified ones (change 2, which ships no DDL — this change owns all of it); registry and memory-scope
unification, including the shadow tree whose importers must be retargeted first (change 3); building the
Cognee memory side and its health probe (change 4). Re-enabling the intentionally commented graph wiring is
out of scope and stays out. `design.md` carries the full Non-Goals list, including the gaps this change
knowingly leaves open.

## Capabilities

### New Capabilities
- `migration-chain-integrity`: one migration head, one authoritative target-schema revision, and the
  guarantee that every relation the running application reads or writes exists after an upgrade.
- `orm-metadata-registration`: every live persisted model visible to the single metadata that schema
  comparison reads, so comparison never proposes dropping a live relation.
- `request-identity-from-token`: the caller's identity derived from validated access-token claims; missing
  credentials answered with 401, never 500.
- `dependency-health-probe`: silently degraded startup dependencies are visible in the health report, and an
  absent optional dependency is reported without failing the overall probe.
- `infrastructure-client-access`: handlers resolve shared clients under the names startup publishes, and every
  database consumer obtains a usable connection URL from a single accessor.

### Modified Capabilities
- `typed-exception-handling`: the requirement *Database operations SHALL catch `asyncpg.exceptions.PostgresError`* is
  restated, and **all six of its scenarios are reproduced verbatim** — including
  `Reconciliation fetch failure catches PostgresError`, whose code this change deletes. It is kept on purpose. A
  `## MODIFIED` block replaces its requirement wholesale on archive, so reproducing five of six would have **deleted**
  the sixth from the deployed spec with no `## REMOVED` block, no Reason and no Migration — and `validate --strict`
  cannot detect that, because the evidence lives in a file this change does not contain. `## REMOVED Requirements`
  is not the alternative: it works at requirement granularity and would retire the whole asyncpg guarantee. Retiring
  the one stale scenario is routed to a direct spec-hygiene edit, alongside the four `## Purpose` failures that also
  need one.
- `transactional-outbox`: two requirements are restated — *Outbox Table Schema*, because the deployed text describes
  a shape the database does not hold, and *Migration*, because the revision it names was stamped rather than applied.
  No requirement is added there; the requirement an earlier draft added, demanding that a missing outbox relation fail
  loudly, is withdrawn because it had no implementing step here and contradicted an accepted requirement in
  `typed-exception-handling` that sanctions the relay's broad catch. That gap is a recorded Non-Goal, and ADR-5 decides
  which spec wins until the narrowing lands.
- `outbox-helper-extraction`: one requirement is restated because the deployed text asserts a property of the auth
  outbox helper that reality only partly satisfies. The correction runs the *opposite* way to the obvious reading: the
  engine-per-call fallback the deployed spec describes still exists and is still reachable from a mounted route, and
  this change does **not** remove it. It is carried as a named outstanding defect against
  `infrastructure-client-access`, owned by the connection-plumbing change.

## Impact

Migration chain and one new revision; the migration environment module; `features/documents`,
`features/search`, `features/ingestion`, `features/agent_saul`, `features/crawler` identity dependencies;
`features/profile` client resolution; `features/health` service, DTO and dependencies (two mounted API
versions); the database connection accessor and its three consumers; `features/__init__`, `tasks/__init__`,
`shared/__init__` and one `pyproject.toml` per-file-ignore key coupled to the deletions.

No data migration: the target relations hold zero rows because they do not yet exist. No new dependency.

## Risks

- **BREAKING 500 → 401** on six mounted document endpoints. This is the point of the change, not a
  side effect; it is called out here so no reviewer meets it in the diff first.
- The health response shape changes on **two** API versions at once, and the checks model forbids unknown
  fields, so the change must be additive and the overall-status computation must be asserted unchanged.
- Deleting 2,900 lines with zero test coverage produces zero test signal. Every deletion is proved by an
  import probe plus an emptiness search plus an unchanged test-failure set — never by "tests pass".
- The migration chain remains partly dishonest after this change: revisions recorded as applied that created
  nothing stay that way, because rewriting them was rejected. `design.md` states the accepted cost and the
  one procedure that gets a fresh environment to the target schema anyway.
- **The chain is branched, so `alembic upgrade head` — singular — does not resolve and exits 255 today.** Three
  committed call sites use the singular form (`Makefile:39`, `README.md:272`, `.github/workflows/test.yml:105`); the
  merge revision repairs all three without editing them. Until then only `heads` is a well-defined target.
- **One of the falsely-applied revisions cannot execute against any database**, which is *why* the stamp happened:
  `9f4a1b7c6d2e` mutates a `clauses` relation that no revision creates and no model declares. That eliminates the
  rewind-and-re-upgrade repair route — it does not terminate — so the repair is a forward, idempotent revision.
  **ADR-6** records the route decision, and the `clauses` question belongs to the search-consolidation change, which
  retargets its readers rather than creating the relation.
- **Applying any of this to the live database is a separately authorized act.** This change writes, renders and
  rehearses the repair against a scratch database; nothing in the sequence assumes the deployed upgrade has run.
- **The outbox relations are the most severe break in the set, and they are on public surface.**
  `POST /auth/forgot-password` and `POST /auth/resend-verification` are mounted, public, rate-limited and
  **return 500 today**, because the event relation they write does not exist — and they fail *after* persisting
  a reset / verification token that no email will ever deliver. That is a partial write on shipped surface, so
  the outbox half of the new revision is ordered first inside it and is justified independently of the
  document schema work.
- **Ordering, load-bearing:** repairing request identity *without* creating the event relations does not repair
  the document upload endpoint — it moves the 500 from the dependency layer down to the event insert. The two
  land together or the repair is illusory.
- The event relay itself fails **soft**, not hard: a broad catch in its startup scan swallows the missing
  relation, so the application boots and the outbox is silently, permanently dead behind two warning lines.
  Boot survives by accident, through an `except` that any tightening pass would remove — which is why
  narrowing that handler is sequenced *after* the relations exist and is not attempted here.
