> Change class: **L** (full checklist + verification matrix).
> Role: reviewer, not author. `proposal.md`, `design.md`, `adrs.md` and both spec deltas were read in full;
> `docs/relay/decisions.md` (D1–D17, D14.1–D14.4), `dispositions.md`, `findings-database.md` §3/§4/§8,
> `findings-openspec-baseline.md` §1 and the four `## Revision` sections of `plan-change2.md` were read as the
> governing record. 18 code claims were re-verified against the tree with `rg`/`sed`; 2 came back wrong.

VERDICT: CHANGES REQUESTED

Seven blocking findings, none of them structural. The change's shape is right — no DDL, no drop, the ADR-first
inversion, three RRF branches, the schema handed upstream — and the spec deltas are mechanically clean (validates
`--strict`, 28/28 scenarios at four hashtags, no leakage). What blocks is: one arithmetic claim that the drift
gate's own rule contradicts (**F1**), one ADDED requirement that surviving code violates with no owner (**F2**),
and two gaps in the ADR that change 1 will code straight into (**F3**, **F4**). The ADR is close to sufficient but
**not yet sufficient** — see F3/F4.

## Completeness

Every disposition-ledger row change 2 owns is present: 184 (retarget, resolution A+) in `design.md` Decision 8,
185 (reader-less derived column, both halves) in Decision 7 + the "Exactly one derived searchable text" requirement.
All four non-mechanical cells of the schema collapse from `dispositions.md:62-66` are addressed — provenance
NOT NULL (Decision 4), chunk `updated_at` (Decision 6), the hardcoded constraint name (Decision 10), the trigram
branch (Decision 7). Item 199 is correctly **not** claimed anywhere in the change. D5.1's "mounting is not part of
this reversal" and D5.2's gating both appear as Non-Goals, and the raw-text ingest loss is recorded three times
(proposal Risks, `design.md` Non-Goals, ADR Consequences) rather than hidden. Gaps: F5 (extensions handed
descriptively, not by name) and N6 (two recorded gaps belong to change 1's ledger row, not change 2's).

## Correctness

F1 is a counting error the gate's own normative rule refutes; F2 is a requirement the code contradicts today; F3
and F4 are ADR omissions measured against "could change 1 get this wrong while following it". F6 and F7 are
statements that survive from the pre-D14.3 framing. Everything else checked out: 16 of 18 code claims verified
exactly, including every column, nullability, default, constraint and index name in `design.md`'s two handoff
tables against `src/app/features/documents/model.py:27-116`.

## Standards

`.opencode/instructions/` axes are not engaged by this change at review time — it authors no code. The two that
will be: the relocated helpers must keep `returns.Result` at the repository boundary (they already do —
`documents/repository.py` returns `AppResult` throughout), and the new `constants.py` must not reintroduce the
hardcoded-SQL pattern Decision 10 rejects. `specs/` is free of class names, function names and library choices
with one exception (N2). `## Purpose` present and substantive on the new capability, correctly absent on the
existing one. Delta headers, four-hashtag scenarios, Reason+Migration on REMOVED: all conformant.

## Risk

The residual risk after fixing F1–F7 is the one `design.md` already names and does not hide: this change's schema
specification ships unexecuted, because nothing in the repository has ever run SQL against a database and the two
target tables have never existed in any environment. The Risks section states that plainly, including the sentence
that makes it auditable later ("If the real-database gate is descoped, this line is the record"). That is the
correct handling. The conftest single-point-of-failure risk (21 module-level imports from the module being
deleted) is correctly mitigated by the re-export shim plus same-commit deletion.

---

## Blocking findings — ranked most severe first

### F1 — The drift gate is red on **one** count today, not three. The gate's own rule says so.

**Where:** `proposal.md:30` ("It is **red on three counts today**"), `design.md:237-238` (Decision 10, same claim),
`design.md:419-420` (step 5's "red before the retarget and green after" proof rests on it).

**Defect.** The gate's normative rule is `specs/document-retrieval-schema/spec.md:159-161`: an identifier must be
*"created by a migration, on a table that a migration creates"*, and
`spec.md:175` forbids the check depending on a reachable database. Measured against that rule:

| Identifier named in query text | Created by | On a table created by | Gate |
|---|---|---|---|
| `clauses_bm25_idx` (`search/repository.py:356,361,362`) | `9f4a1b7c6d2e:132` | **nothing** — `9f4a1b7c6d2e:28` creates `parent_documents`; `clauses` is only `batch_alter_table`d at `:63` | **RED** |
| `search_chunks_bm25_idx` (`search/repository.py:415,417,419,430`) | `8a7d9b1c2e3f:86` | `8a7d9b1c2e3f:44` | **GREEN** |
| `uq_search_chunks_document_chunk_index` (`search/repository.py:157`) | declared in `8a7d9b1c2e3f:70` `create_table` | `8a7d9b1c2e3f:44` | **GREEN** |
| `chunks_bm25_idx` (`documents/repository.py:323,327,331,533-540`) | `a71f0d7d9c12:97` | `a71f0d7d9c12:52` | GREEN |
| `uq_chunks_document_chunk_index` (`documents/repository.py:222`) | declared in `documents/model.py:74` | `a71f0d7d9c12:52` | GREEN |

"Three" comes from `plan-change2.md:898-907` (B3 revised), which counted objects that *do not exist in the live
database* — a criterion the spec explicitly excludes. This produces wrong work: the implementer writes the gate,
sees one red, and must either make it database-aware (spec-forbidden), pad it with an expected-fail list, or
conclude the gate is broken and weaken it.

**Fix.** Restate as red on **one** count (`clauses_bm25_idx`, an index created on a table no migration creates) in
all three places. Step 5's red-before/green-after proof still works unchanged — the retarget at step 4 is exactly
what removes that one reader. If a stronger gate is wanted, that is a second rule ("no query names a table no
migration creates", already `spec.md:35-38`) and should be counted separately, not folded into this one.

### F2 — An ADDED requirement is violated by the surviving code, with no task and no Non-Goal.

**Where:** `specs/document-retrieval-schema/spec.md:106-110`:

```
#### Scenario: A mode's required database capability is absent
- **WHEN** the database does not provide an extension or index a retrieval mode requires
- **THEN** provisioning SHALL fail loudly with that missing capability named
- **AND** the system SHALL NOT silently serve a fused result from fewer modes than it declares
```

**Defect.** `src/app/features/documents/service.py:294-300` — the code this change keeps and declares the sole
retrieval truth — does exactly what the scenario forbids:

```python
for r in results:
    if isinstance(r, Success):
        row_sets.append(_to_ranked_rows(r.unwrap()))
    elif isinstance(r, Failure):
        log_expected_failure(r.failure(), operation="hybrid_search")
        row_sets.append([])
```

A branch whose extension is missing returns `Failure`, contributes an empty rank list, and the request returns 200
with a fused result silently built from fewer modes. This is not hypothetical: **D14.4 leaves F8 open** — whether
`pg_textsearch` registers an access method literally named `bm25` was never confirmed — so `to_bm25query` failing
at runtime is the expected first encounter with this path. `design.md`'s Non-Goals do not disclaim it and the
source-change order (steps 1-10) contains no task for it. Whoever writes `tasks.md` has a requirement with no task.

Second, smaller defect in the same scenario: "provisioning" appears nowhere else in the change and is undefined —
it is change 0's migration, which this spec cannot name.

**Fix.** Pick one and make it explicit: (a) add a step to `design.md`'s source-change order that makes a
branch-level `Failure` in `search_documents` fail the request rather than degrade, and keep the scenario; (b) narrow
the scenario to the provisioning half only (drop the runtime `AND`, and say the missing-capability check is change
0's `CREATE EXTENSION`); or (c) record silent branch degradation as a Non-Goal owned by change 1 alongside the
re-ranking and floor-calibration work, and delete the runtime clause. (a) or (c) are consistent with the rest of the
change; (b) leaves a requirement about an undefined actor.

### F3 — ADR: `id` generation is unspecified. Change 1 writes chunks by raw SQL and will hit NOT NULL.

**Where:** `adrs.md:44-62` (documents required fields + "everything else is nullable or defaulted") and
`adrs.md:64-78` (chunks). `id` appears in **neither** list. `design.md:343` and `:366` give `id` the Default cell
"primary key", which is not a default, and the reference DDL confirms there is none: `a71f0d7d9c12:30`
`sa.Column("id", sa.UUID(), nullable=False)`.

**Defect.** The ADR is the sole artifact change 1 reads (D15, `adrs.md:9-13`). Its contract is stated as a closed
set: "a writer MUST supply all four/all five of…", then "everything else is nullable or defaulted". `id` is in
neither bucket, so a literal reading says change 1 need not supply it — and change 1's persistence nodes write by
raw `text()` INSERT today (`shared/langgraph_layer/ingestion_kb/nodes.py:495-514` supplies no `id`;
`:551-560` supplies an explicit one), where SQLAlchemy's Python-side `default=uuid4` never fires. That is a
`NotNullViolation` on first write, discovered at change 1 implementation time, caused by following the ADR.

**Fix.** Add one row to each table's contract in `adrs.md`: `id` — uuid, NOT NULL, **supplied by the writer**
(ORM default `uuid4`; no database default exists, so a non-ORM writer must generate it), and say so alongside the
`search_text` "MUST NOT be supplied" note so both directions are pinned. If a `DEFAULT gen_random_uuid()` is wanted
instead, that is a change-0 DDL input and must appear in `design.md:343`/`:366` rather than being left implicit.

### F4 — ADR: the column set is closed with no home for the parsed text, summary or thread scope change 1 persists.

**Where:** `adrs.md:28-40` (Decision + Table identity: "Exactly two tables"; `adrs.md:126-131` excludes a third
table for hierarchy) versus what change 1's promoted pipeline writes today,
`shared/langgraph_layer/ingestion_kb/nodes.py:496-499`:

```sql
INSERT INTO parent_documents
    (doc_id, user_id, thread_id, source, title, document_type, jurisdiction,
     content_hash, markdown, summary, metadata)
```

**Defect.** Mapping that row onto the ADR's `documents`: `document_type`→`document_kind`, `source`→`source_uri`,
`jurisdiction`/`title`/`content_hash`/`user_id` map directly. **`markdown` (the full parsed document text),
`summary` and `thread_id` have no column and no instruction.** The ADR's stated rationale for `object_uri` ("the
provenance link that makes re-parsing after a chunker change possible", `adrs.md:49`) strongly implies parsed text
is *not* stored and is re-derived — but it never says so, and the ADR does offer `metadata_` jsonb as the catch-all
for "structure carried on the row". An implementer following it can reasonably put a multi-megabyte markdown
string into a jsonb column, and `ON CONFLICT (doc_id)` at `nodes.py:503` also has to be re-pointed at
`uq_documents_user_content_hash` — which the ADR does pin, correctly.

**Fix.** Add to `adrs.md`'s "Explicitly NOT part of this contract": the parsed document text is **not** stored on
`documents` — `object_uri` is the provenance link and parsed text is re-derived from the stored object; `summary`
and any thread/session scope belong in `metadata_`, and `metadata_` is for scalars, never for document bodies. Two
sentences, and they remove the only decision change 1 would otherwise have to invent.

### F5 — `design.md` hands change 0 four extensions descriptively, never by name, and does not pin the vector index method.

**Where:** `design.md:396-400` — *"Extensions the migration must create itself, not inherit. Vector support,
approximate-vector indexing, keyword search and character similarity."* And `design.md:393` —
*"`chunks_embedding_idx` — approximate vector search over `embedding` with cosine distance."*

**Defect.** D14.4 (`decisions.md:391-394`) requires change 0 to create **all four extensions explicitly**, precisely
because ambient availability is luck. The handoff names none of them, and the count is only inferable. The live
server (`findings-database.md:75-86`) has *two* candidates for "approximate-vector indexing": `vectorscale`
(installed, provides `diskann`) and `vchord` (available, does not). The index spec omits the access method, so
change 0 can satisfy this handoff literally and produce an index the reference revision's `USING diskann`
(`a71f0d7d9c12:100`) does not match, or create `vchord` and have the index fail. `design.md` is the right venue for
these names — library choices are permitted here and barred only in `specs/`.

**Fix.** Name them: `vector`, `vectorscale`, `pg_textsearch`, `pg_trgm`, all four `CREATE EXTENSION IF NOT EXISTS`
in the authoritative revision. Pin `chunks_embedding_idx` as `USING diskann (embedding vector_cosine_ops)` (or state
explicitly that the access method is change 0's choice and `vectorscale` is therefore not required). The
`diskann`-survives-only-by-luck note at `:397-400` is correct and verified — `a71f0d7d9c12:23-26` creates
`uuid-ossp`, `vector`, `pg_trgm`, `pg_textsearch` and **not** `vectorscale`, while `8a7d9b1c2e3f:26` is the only
revision that creates it — but the note is useless to change 0 without the names.

### F6 — A requirement **title** carries the unreachable from-base claim D14.3 retired.

**Where:** `specs/document-retrieval-schema/spec.md:196` — `### Requirement: A provisioned database contains only
the unified stores`.

**Defect.** The requirement body and both scenarios were correctly re-scoped to *"the project's authoritative schema
migration"* — that is exactly D14.3's binding restatement, and no `alembic upgrade head --sql` from-base proof
survives anywhere in the change (grepped: zero hits). But the **title** asserts the from-base outcome, and it is
false in both directions: a fresh `alembic upgrade head` traverses `8a7d9b1c2e3f` and creates
`search_documents`/`search_chunks`, then fails at `9f4a1b7c6d2e:103`'s ALTER of the phantom `clauses`.
`design.md:265-273` says this in so many words. Requirement titles are the durable key an archived spec is read and
`MODIFIED`-matched by, so a false title outlives the change.

**Fix.** Rename to what the body actually says, e.g. `### Requirement: The authoritative schema migration creates
only the unified stores`. Body and scenarios need no edit.

### F7 — The mounted-owner requirement imports change 0's `UserIdDep` fix without recording the dependency.

**Where:** `specs/document-retrieval-schema/spec.md:180-193` (`Requirement: Mounted document surface requires an
authenticated owner`, scenario 2: *"the refusal SHALL be an explicit authorization failure, not an unhandled
internal error"*) versus `design.md:79-80`, which makes fixing owner resolution an explicit **Non-Goal**, and
`adrs.md:139-141`, which excludes reachability and authorization from the contract entirely.

**Defect.** Scenario 2 is D5.2's fix verbatim (`decisions.md:145-150`): `documents/dependencies.py:61-62` reads
`request.state.user_id` unguarded, so today the mounted endpoint raises `AttributeError` — an unhandled internal
error, exactly what the scenario forbids. Change 2 writes no code for it. It becomes true only because change 0
lands first, and `design.md:436-439`'s Coordination points list four change-0 deliverables (head merge, create-schema
migration, task-registry rewrite, model registration) and **not** the `UserIdDep` fix. So the one cross-change
precondition this requirement depends on is the one not written down. Scenario 1 ("the mounted route set gains
nothing") is change 2's own and is sound — verified: `features/search/router.py` has no importer anywhere in `src/`
or `tests/`, and `api/v1.py:12-17` mounts six routers, none of them search's.

**Fix.** Add the `UserIdDep` fix to `design.md`'s Coordination points as a hard precondition for this requirement
(one line, naming D5.2), or drop scenario 2 and keep the requirement to the no-new-reachability invariant change 2
can prove by route enumeration.

---

## Non-blocking nits

- **N1 — "twenty symbols" is 21.** `proposal.md:77`, `design.md:242`, `design.md:251-252`. `tests/conftest.py:56-83`
  imports 2 (chunking) + 5 (constants) + 6 (dto) + 3 (fusion) + 3 (rag) + `SearchRepository` + `SearchService` = **21**.
  The argument is unaffected.
- **N2 — one library name in `specs/`.** `specs/document-retrieval-schema/spec.md:215`: *"no model, query, Celery task
  or test fixture"*. "Celery" is a library choice; "background task" is the behaviour-contract phrasing.
- **N3 — `design.md:399` contradicts `design.md:39-43`.** *"the index survives today purely because the extension
  happens to be installed already"* — the index has never been created, as the same document states four
  paragraphs earlier. The phrasing is inherited from D14.4 (`decisions.md:392-394`); restate as "the declaration
  would build only because `vectorscale` happens to be pre-installed".
- **N4 — the handoff tables imply server defaults the ORM does not carry.** `design.md:349-354` and `:370-380` give
  `'generic'`, `'received'`, `'[]'`, `'{}'`, `0`, `false` as Defaults, while `documents/model.py:45-50,89-99` declares
  them Python-side only (`default=`, not `server_default=`). Harmless now (step 9 gate 2 is deferred), but it is
  guaranteed `alembic check` drift the moment that gate becomes usable. Say which side is authoritative.
- **N5 — `proposal.md:12-13`'s superset claim is not exact.** Search's `/ingest/{task_id}` (`search/router.py:39`) is
  a *task*-status read with no documents equivalent — documents has `/documents/{doc_id}/status`
  (`documents/router.py:50`), keyed by document. Immaterial, since the ingest that endpoint tracked is deleted, but
  "every read endpoint" is stronger than the tree supports.
- **N6 — two recorded gaps in `design.md:97-101` belong to change 1's ledger row.** `dispositions.md:43` (138
  residue b, `vector_store` singleton, DROP) and `:50` (165 + Up#3, external RAG frameworks, DEFER) sit in the
  **change 1** table. Carrying them here is harmless, but D13 puts a gap in *the owning change's* Non-Goals — the
  hazard is change 1's design omitting them because change 2 appears to have covered them. `dispositions.md:47`
  (163, MERGE → 185) genuinely is change 2's.
- **N7 — `MODIFIED` adds a scenario.** `specs/llm-injection/spec.md:19-22` adds *"No dependency layer survives for
  the dissolved search service"* to a `MODIFIED` requirement. Permitted (the whole block replaces the original), but
  the conventions file prefers `ADDED` for a new concern. Related: `proposal.md:57` says the documents half is kept
  "unchanged" while the text was in fact rewritten — correctly, to strip the `search/dependencies.py` /
  `_build_chat_model()` file-and-function names out of a spec. Reword the proposal, not the delta.
- **N8 — no coordination line for the relocated chunker.** `design.md:413-414` moves `search/chunking.py` into the
  documents feature; D8 gives change 1 "hierarchical chunking for legal docs", which may replace it outright. Worth
  one line in Coordination points so change 1 does not re-relocate or orphan it.

## Verified clean

Distinguishing "verified good" from "not examined":

- **Mechanics.** `openspec validate documents-unified-schema --strict` → *"Change 'documents-unified-schema' is
  valid"*. `openspec validate --all` → **21 passed / 6 failed**, i.e. the same six pre-existing failures as the
  D12 baseline (`change/mintlify-documentation`, `spec/cognee-v1-api`, `spec/noqa-documentation`,
  `spec/pattern-matching-standard`, `spec/transactional-outbox`, `spec/typed-exception-handling`) — **no new
  failures**, consistent with `findings-openspec-baseline.md` §1.
- **Delta shape.** 12 requirements / 28 scenarios. Every scenario header is exactly four hashtags (grepped for 3-hash
  `Scenario` headers: zero). Every requirement has ≥1 scenario. Operation headers are `## ADDED Requirements`,
  `## MODIFIED Requirements`, `## REMOVED Requirements` only.
- **`llm-injection` delta (check 6).** `### Requirement: Dependency Layer Updates` matches
  `openspec/specs/llm-injection/spec.md:20` verbatim; the original's **single** scenario
  (`Dependency layer creates LLM once and injects`, `:24`) is reproduced by name, so no original scenario is lost.
  `### Requirement: SearchService Constructor Injection` matches `:5` verbatim and carries **both** `**Reason**` and
  `**Migration**`. No `## Purpose` on this existing capability (correct — a Purpose on a delta is ignored, and the
  deployed one is a `TBD` stub that only a direct file edit can fix). The two untouched requirements
  (`Document Service Injection`, `Backward Compatibility`) are correctly absent from the delta. The new capability's
  `## Purpose` is real and substantive, not a stub.
- **Check 2 — no DDL, no drop.** Zero `DROP TABLE`/`drop_table`/`DROP INDEX` anywhere in the change (the single
  match is `design.md:132` rejecting a `DROP TABLE IF EXISTS` revision as an alternative). Nothing in either spec
  mandates change 2 author a migration; the migration-facing requirements constrain "the project's authoritative
  schema migration", which is change 0's under D14/D14.1.
- **Check 3 — the from-base proof is gone.** Grep for `alembic`, `upgrade head`, `--sql`, `grep -c` across the whole
  change directory returns one hit, `design.md:6`'s `alembic_version`. `design.md:265-273` states the limit
  explicitly and does not claim zero. The replacement (the authoritative revision's own rendering, plus a source-tree
  scan outside migration history) is checkable by `alembic upgrade <down>:<rev> --sql` and `rg`.
- **Check 4 — D16.** Specified as a contract (`spec.md:141-155`, "written by the ingestion path itself, not left to
  an update-only mechanism that a conflict-resolving insert bypasses"), pinned in the ADR as a writer-supplied
  required field with the trap spelled out (`adrs.md:73,88-93`), handed to change 0 as a `CREATE TABLE` column
  (`design.md:383`, "new column, in the `CREATE TABLE`, never an `ALTER`"), and the ORM/conflict-set/row-builder edit
  kept in change 2 (`design.md:431-432`). Correct on every axis. The trap is real and verified: the conflict set at
  `documents/repository.py:222-233` omits `updated_at`, and `build_chunk_rows` at `:601-604` does not add it.
- **Check 5 — three branches.** Reflected in `proposal.md:35`, `design.md:32-37`, `design.md:190-201` (Decision 7,
  with the two-branch fallback kept on the record), `spec.md:82-110`, and the index list at `design.md:389-394`.
  The extension-name gap is F5; the count and the decision are right.
- **Check 9 — scope.** No claim on change 0's DDL, extensions, `UserIdDep` (F7 is a missing precondition note, not a
  claim of the work) or outbox; none on change 1's ingestion graph, embedder or embedding decisions — the embedding
  client is explicitly left in place (`design.md:414`); none on change 3's tool retargeting (`design.md:439`).
  **Item 199 is not claimed** (grepped: zero mentions of `DocumentQueryService` or `object | None` in the change),
  and it is indeed already fixed in-tree — `documents/service.py:236-237` reads `redis: Redis | None`,
  `graphiti: Graphiti | None`.
- **"No endpoint becomes newly reachable" — verified true.** `features/search/router.py` has no importer in `src/` or
  `tests/`; `api/v1.py:12-17` mounts auth, health, users, profile, documents, agent_saul. The documents router's six
  paths are untouched by this change, and the moved graph-backed ask path is left unexposed (`design.md:421`).

### Code claims spot-checked (18; 16 confirmed, 2 wrong)

| # | Claim | Result |
|---|---|---|
| 1 | `search/repository.py:157` `constraint="uq_search_chunks_document_chunk_index"` | confirmed |
| 2 | `search/constants.py:15` BM25 index-name constant; `:8` `RRF_K = 60` | confirmed |
| 3 | `search/model.py:75-79` `content_tsv` STORED generated; only non-source reader is `8a7d9b1c2e3f:53,94` → zero readers | confirmed |
| 4 | every column/null/default/constraint/index in `design.md`'s two tables vs `documents/model.py:27-116`, and `UnifiedChunk` has no `updated_at` | confirmed |
| 5 | `documents/repository.py:222` conflict set and `:601` row builder both omit `updated_at` | confirmed |
| 6 | `documents/repository.py:323,327,331,533-540` name `chunks_bm25_idx` inside query text; `:406-428` trigram via `c.search_text % :query` | confirmed |
| 7 | `a71f0d7d9c12:97,100,103` create the three retrieval indexes; `:23-26` create four extensions but **not** `vectorscale`; `8a7d9b1c2e3f:26` is the only revision creating it | confirmed |
| 8 | no migration creates table `clauses` (`9f4a1b7c6d2e:28` creates `parent_documents`; `:63` alters `clauses`; `:132` indexes it) | confirmed |
| 9 | `search/repository.py:308` `legal_rrf_search` → `FROM clauses` (`:337`), `clauses_bm25_idx` (`:356,361,362`), `JOIN clauses` (`:383`) | confirmed |
| 10 | `retrieval_kb/nodes.py:172-193` hybrid node calls `repo.legal_rrf_search`; `build_retrieval_graph`'s only caller is `search/service.py:259` | confirmed |
| 11 | `search/service.py:68` `self._llm`; `:106` `with_outbox` inside `ingest_document`; `:257-260` `ask_legal` → `build_retrieval_graph` | confirmed |
| 12 | `documents/service.py:520` and `:686` — every chunk row upserted twice per ingest | confirmed |
| 13 | `tests/conftest.py` imports **twenty** symbols from the module being deleted | **wrong — 21** (N1) |
| 14 | `connections/celery.py:194` includes `tasks.search_tasks` | confirmed |
| 15 | stale `__pycache__` entries with no source: `dependency.cpython-312.pyc`, `handler.cpython-312.pyc` | confirmed |
| 16 | `tests/conftest.py` has no engine, no `create_all`, no container; `tests/integration/test_search.py` mocks repo, embedding client and session | confirmed |
| 17 | the drift gate is **red on three counts** today | **wrong — one** (F1) |
| 18 | `documents/` imports helpers out of `search/` and never the reverse (`documents/dto.py:7`, `service.py:15,86`, `repository.py:15`) | confirmed |

## Verdict

The ADR is **not yet sufficient** for change 1 to implement against — it is close, and its identity keys, dedup key,
upsert key, generated-column prohibition, nullability table and exclusion list are all correct and unusually
well-pinned, but F3 (`id` generation absent from a contract stated as a closed set) and F4 (no home for the parsed
text, summary and thread scope change 1's pipeline persists today) are both defects an implementer hits on first
contact while following it faithfully. Both are two-sentence fixes. Under D15 the ADR is accepted before change 1
codes, so these must land before change 1 starts, not with change 2's code.

F1, F2 and F5 must be fixed before `tasks.md` is written: F1 and F2 would each produce a task whose Proof cannot
pass as stated, and F5 leaves change 0's migration under-specified in exactly the way D14.4 was recorded to prevent.
F6 and F7 are one-line restatements. The eight nits do not block.

VERDICT: CHANGES REQUESTED

---

## Author response

Every finding was re-verified against the tree before acting. None was refuted; F1 was confirmed **and extended**.

- **F1: fixed (+ extended).** Confirmed: counted under the gate's own rule, only `clauses_bm25_idx` is red. No
  revision creates table `clauses` (`9f4a1b7c6d2e` only `batch_alter_table`s it at `:63`, `alter_column`s it at
  `:103-105`, indexes it at `:115-138`); `search_chunks` is created at `8a7d9b1c2e3f:45`, and both
  `search_chunks_bm25_idx` (`:86`) and `uq_search_chunks_document_chunk_index` (`:67-71`, inside the `create_table`)
  are created by that same revision — green. "Three" was a live-database criterion the spec forbids. Corrected in
  all three places, and `design.md` Decision 10 now carries the five-identifier table with the arithmetic shown and
  the reason the wrong count would have produced unpassable work. **Extension the review missed:** the one red
  identifier has **two** source readers, not one — `shared/langgraph_layer/ingestion_kb/nodes.py:751` also names it,
  in `SELECT bm25_force_merge('clauses_bm25_idx')`. So the retarget at the old step 4 would **not** have turned the
  gate green, and step 5's red-before/green-after proof would have failed as written even after the count was fixed.
  The change now retargets that single string literal too (recorded in `design.md`'s source-change order, its
  Coordination points and the proposal's Impact list as the one line this change touches in change 1's module), and
  `tasks.md` step 9 proves red-before/green-after against both readers.
- **F2: fixed (code task added, requirement kept and strengthened).** Confirmed at
  `documents/service.py:294-300` — a branch `Failure` is logged and contributes `[]` to the fusion. Per the
  coordinator's ruling on change 1's finding C, fail-loudly stands and this change owns it, so the scenario was
  **not** softened and no weakening edit was made at any point. Resolved as the review's option (a): a new
  source-change step (now step 5) makes a branch failure fail the request with the branch named, a matching Goal,
  and `tasks.md` step 8 with an executable proof. The requirement gained one scenario, *One mode fails to execute*,
  which also pins the distinction the code fix turns on — an empty result from a healthy branch is not a failure.
  The undefined actor "provisioning" is now defined in `design.md` (change 0's `CREATE EXTENSION`/`CREATE INDEX`
  statements) rather than removed from the spec, since the specs may not name a change.
- **F3: fixed.** Confirmed: `a71f0d7d9c12:30` and `:54` declare `id uuid NOT NULL` with no server default; the ORM
  default is Python-side (`documents/model.py:39,81` `default=uuid4`), which does not fire for the raw `text()`
  inserts change 1's pipeline uses. `id` is now a required-writer row on **both** tables in `adrs.md` (four→five,
  five→six), stating that the writer generates it and that no database default exists, with a paragraph pinning
  `id` MUST-supply and `search_text` MUST-NOT-supply together, and naming `DEFAULT gen_random_uuid()` as a change-0
  DDL input if wanted instead. `design.md`'s two tables no longer print "primary key" in the Default cell.
- **F4: fixed.** Confirmed at `ingestion_kb/nodes.py:496-499` — `markdown`, `summary`, `thread_id` are persisted and
  had no home. `adrs.md`'s exclusion list now states that `documents` stores **no** full-text body column, that
  `object_uri` is the provenance link and parsed text is re-derived from the stored object, that `summary` and any
  thread/session scope go in `metadata_` as scalars, and that `metadata_` is never a home for a document body. The
  `ON CONFLICT (doc_id)` trap at `nodes.py:503` is closed in the same pass: the identity-keys section now says there
  is no `doc_id` column to conflict on.
- **F5: fixed.** Confirmed: `a71f0d7d9c12:23-26` creates `uuid-ossp`, `vector`, `pg_trgm`, `pg_textsearch` and not
  `vectorscale`; `8a7d9b1c2e3f:26` is the only revision creating it; `a71f0d7d9c12:100` uses `USING diskann`. All
  four extensions are now named in a table with what each provides and its live state, and `chunks_embedding_idx`
  is pinned to `USING diskann (embedding vector_cosine_ops)` with `vectorscale` named as its source and `vchord`
  named as the candidate that would not satisfy it.
- **F6: fixed.** Retitled to `### Requirement: The authoritative schema migration creates only the unified stores`.
  Body and scenarios unchanged. Free of the `MODIFIED` verbatim-reproduction trap: the requirement is `ADDED` on a
  new capability, so there is no archived title to match.
- **F7: fixed (precondition recorded).** Confirmed: `documents/dependencies.py:61-62` reads `request.state.user_id`
  unguarded. Scenario 2 kept, and `design.md`'s Coordination points now name change 0's `UserIdDep` fix (D5.2) as a
  hard precondition for it, state that this change writes no code for it and cannot, and separate scenario 1 as the
  half provable by route enumeration without change 0.

**Nits: all eight addressed, none skipped.** N1 twenty→twenty-one in all three places (re-counted: 2 + 5 + 6 + 3 + 3
+ `SearchRepository` + `SearchService` = 21). N2 "Celery task"→"background task". N3 restated as the *extension*
being inherited by luck, with the never-created index called out so the two statements no longer collide. N4 the
ORM is declared authoritative for the application-side defaults, with `created_at`/`updated_at`/`search_text` named
as the deliberate DDL-side exceptions and `server_default` prohibited for the rest. N5 the superset claim now names
the task-status exception. N6 the two change-1 gaps are marked as remaining change 1's to record, with the
change-2-owned one identified. N7 the proposal now says the surviving half was restated rather than "unchanged";
the delta stays `MODIFIED`, which the reviewer confirms is permitted. N8 a Coordination point tells change 1 to
replace the relocated chunker in place.

**Also changed:** a Coordination point records that change 1 defers to `document-retrieval-schema` for
extension-missing behaviour and for fusion, per the coordinator's ruling.

**F2 (from change 3's review, routed by orchestrator): fixed — accommodated in full, no new table.** Change 3's
`legal-corpus-retrieval` requires statute identity attributes to be addressable and the
(instrument, section) point lookup to be index-served, and only this change can deliver it because change 3 ships no
DDL. Evaluated on the attribute-level language rather than the absent `act_name`/`section_ref` strings. The column
set is now **open to it and closed around it**: `chunks.instrument_name` (varchar(255), NULL — normalized, and
deliberately not the parent document's free-text `title`, because identity that piggybacks on a display string
drifts the first time a title is reformatted), `chunks.section_ref` (varchar(64), NULL, stored as authored and
compared whole), `chunks.instrument_year` (smallint, NULL, the vintage discriminator), plus
`ix_chunks_instrument_section` on `(user_id, instrument_name, section_ref, instrument_year)`, **partial**
`WHERE instrument_name IS NOT NULL`. The column order is the design: tenant leads because every read in this schema
is tenant-scoped; the two identity columns give a one-descent point lookup; the year trails so a backward btree scan
yields *newest applicable year first* with no sort, aggregate or self-join; and the partial predicate keeps the
non-statute majority of chunks off the index, which matters on the hottest table in the schema. The
newest-applicable-year rule is written out with its NULL and tie behaviour so two readers implement it identically.
A statutory provision is therefore **a chunk, not a table** — the never-created `statutes` table stays uncreated,
and the ADR's third-table exclusion now says so explicitly. Recorded in `adrs.md` (new subsection plus the
nullable list plus the exclusion bullet), handed to change 0 in `design.md`'s chunk column table and index list, and
recorded as a Coordination point so change 3 references this contract instead of re-specifying it. Deliberately **no
new requirement in `document-retrieval-schema`**: change 3's requirement already governs the behaviour, and
duplicating it here would recreate exactly the two-owners-one-code-path conflict the fusion ruling just resolved.
`tasks.md` step 15 carries the ORM half.

`tasks.md` written: 16 steps, each with an executable Proof. No proof renders migrations from base (D14.3); the two
that touch migration output scope to the authoritative revision's own rendering. Steps depending on change 0 or
change 1 name the dependency.
