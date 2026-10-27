# PLAN — Bulletproof Ingestion First

**Date:** 2026-10-27 · **Owner:** Harmeet · **Status:** ready to execute
**Hand-off doc:** self-contained. Give this file to any agent session; it re-derives nothing.

---

## 0. What this plan is

**North star (chosen):** *bulletproof ingestion* — the live
upload → parse → chunk → extract → persist → graph-write pipeline never lies,
never loses data silently, and never degrades without a typed signal.

**Constraints (chosen):**

- Pre-launch: the only affected user is you. Aggressive moves are fine; there is
  no deprecation window and no staging requirement.
- **Nothing gets deleted.** Disconnected surfaces are *quarantined* (unmounted,
  documented, revisited later), never removed in this plan.
- Sequencing must not conflict with the in-flight OpenSpec change
  `openspec/changes/harden-agent-saul-evidence-pipeline/` (86 tasks, **0 done**
  as of today). Where that change already specs a piece of work, this plan
  points at its task numbers instead of re-specing it.

**What "bulletproof" means here, concretely — three properties:**

1. **Truthful** — every terminal state matches what actually happened. No
   success logs for work that never ran, no retryable errors on permanent
   failures, no docstrings describing a world that isn't on disk.
2. **Bounded** — a hostile or oversized input is rejected or budgeted *before*
   it costs money or memory, not after.
3. **Observable** — when fidelity is degraded (bad OCR, mixed contracts,
   dropped tables), the document record carries that fact forward instead of
   flushing it in a log line.

---

## 1. Verified state of the tree (2026-10-27)

Everything below was re-verified on disk today, against `main` at `2229cc1`.

| # | Surface | Path | Status | Proof |
|---|---------|------|--------|-------|
| A | Documents pipeline (the live one) | `src/app/features/documents/` + `src/app/extraction/langextract/` + `src/app/retrieval/graphiti/` | **LIVE** | `POST /documents/upload` → Celery → `service/ingest.py` |
| B | `ingestion_kb` LangGraph | — | **DELETED since Oct 11** | commit `3bd67da` "delete dead trees"; `build_ingestion_graph` has zero references in `src/` |
| C | `features/ingestion` HTTP | `src/app/features/ingestion/` | **LIVE, MISWIRED → guaranteed 503** | mounted at `api/v1/__init__.py:19` and `api/v2/__init__.py:42` while `service.py:59-71` still sends a `doc_id`/`raw_bytes` state shape that no live graph declares |
| D | Docling ingest_v2 | `src/app/extraction/docling/ingest_v2.py` | ORPHAN (tests only) | zero production importers |
| E | LangExtract batch processor | `src/app/extraction/langextract/langextract_batch_processor.py`, `docling_preprocessor.py` | ORPHAN | zero importers |
| F | Retrieval graph | `src/app/features/documents/retrieval_graph/` | **BUILT, DELIBERATELY UNMOUNTED** | `query.py:584` `ask_via_retrieval_graph` docstring: "Deliberately not exposed by any router"; mounted `/search/ask` + `/legal/ask` still run the inline duplicate in `ask` |
| G | Graph relationship writer | `src/app/retrieval/graphiti/write_clause_episodes.py` | **LIVE BUT FED `[]`** | `service/ingest.py:738-740` passes `[]`; the legal knowledge graph has nodes and zero edges |
| H | Agent memory-write tools | `src/app/features/documents/tools/bundle.py:72-73` | UNWIRED | docstring still says "implemented but not yet wired into any node" |

**Known false prose still on disk** (reconciled in Phase A):

- `src/app/features/ingestion/dependencies.py:10-25` — claims the lifespan
  assignment "is commented out by decision (D17)" and that "the router that
  would consume this is not mounted … so no unavailable surface actually ships
  today." Both false: `lifecycle/lifespan.py:328-347` provisions
  `app.state.ingestion_graph`, and the router is mounted twice (C above).

**Baseline gates** (from `AGENTS.md` — run all four after every phase):

```bash
uv run ruff check --fix src/
uv run ty check src/
uv run pytest
ast-grep scan src/
```

Record the baseline results *before* Phase A touches anything, so pre-existing
failures are never attributed to the plan.

---

## 2. Strategy

The Oct 10–11 analysis framed the cluster as "delete vs harvest." Since then,
`3bd67da` already deleted the big dead tree (`ingestion_kb`, ~1,000 lines), so
the only real disconnected *surfaces* left are **C** (the lying endpoint) and
**F** (the unmounted retrieval graph). "Don't delete for now" is therefore
cheap: the keep-list is one miswired service and one dormant graph.

The order below is the north-star filter applied to what's left:

1. **A lying endpoint on the ingestion boundary is a bulletproof-ingestion
   defect**, not cleanup. Unmount it first. (Phase A)
2. **The pipeline must tell the truth before it can be hardened** — you cannot
   gate fidelity you currently report as success when nothing ran. (Phase B)
3. **Then harden**: fixtures → classification boundaries → fidelity gates →
   budgets. This is the body of "bulletproof ingestion" and it is already
   specced in the harden change; this plan sequences it and marks the entry
   tasks. (Phase C)
4. **Graph edges (todo 136) come after**, because edges multiplied by
   untrustworthy ingestion just makes confident garbage — and because Phase B
   already put the truthful signal in place that Phase D fills. (Phase D)
5. **Retrieval unification and agent tools** stay delegated to the harden
   change (its Phases 5–6), with the one rule stated here: *one implementation
   of the plan→retrieve→grade→generate sequence, never two*; and *no graph
   tools for the agent until the graph has edges*. (Phase F)

Each phase ends at a green gate and a commit. No phase requires a decision
from a later phase.

---

## 3. Phases

### Phase A — stop the lying surface (~30 min, 2 lines + docstrings)

The endpoint `POST /ingestion/documents/upload` resolves path A's graph,
sends it path B's state keys, invokes without config, and returns an HTTP 503
with `retryable=True` on **every** call — after buffering the whole upload.
Unmount; do not delete (per constraint).

**Files:**

- `src/app/api/v1/__init__.py` — delete line 19
  (`v1_router.include_router(ingestion_router)`) and the import at line 10.
- `src/app/api/v2/__init__.py` — delete line 42 and the import at line 20.
- `src/app/features/ingestion/dependencies.py` — rewrite the docstring
  (lines 10–25) to state the new truth, e.g.:
  > *Unmounted from v1/v2 on 2026-10-27: this router was wired to the documents
  > graph but sent an incompatible state shape, so every call was a guaranteed
  > 503. The service and dependency are retained (quarantined — see
  > `docs/quarantine-ledger.md`) pending a deliberate wiring decision. If ever
  > reached, this dependency still fails closed with a typed 503.*

**Acceptance:**

- `rg -n "include_router\(ingestion_router\)" src/app/api/` → no hits.
- `uv run pytest` — expected to break nothing: as of the Oct 10 verification
  no test asserts the mount exists (the old
  `test_unprovisioned_graph_fails_closed.py` no longer exists under that name;
  re-confirm with `rg -l "ingestion_router" tests/` — if a test does assert the
  mount, that test is documenting the defect and should be updated, not the
  mount restored).
- Manual: `POST /v1/ingestion/documents/upload` → 404 (not 503).
- All four gates green.

**Commit:** `fix(ingestion): unmount miswired ingestion router (guaranteed 503)`

### Phase B — make the graph-write truthful (vacuity fix, ~1 hour)

`service/ingest.py:738-756` passes `[]` to
`write_clause_episodes_to_graphiti` and then computes
`all(result.success for result in [*clause_results, *relationship_results])`.
Since `all([]) is True`, "zero relationship writes attempted" is
indistinguishable from "all relationship writes succeeded" on every
observability surface. This is the exact defect class "bulletproof" exists to
kill.

**Files:**

- `src/app/features/documents/service/ingest.py` — the graph-write mapping
  function (~lines 695–756).

**Change:**

1. Success is computed over `clause_results` only.
2. When relationship extraction has not run, emit one structured log/event:
   `graph_relationships_skipped` with
   `{relationships_attempted: 0, reason: "relationship_extraction_not_implemented"}`.
   Distinct name, so dashboards count it rather than reading a success line.
3. Keep the existing failure path for `relationship_results` (Phase D will
   populate the list and this path starts earning its keep).

**Acceptance:**

- New test: with the current (empty) relationship path, the emitted log is
  `graph_relationships_skipped`, and the function's return reflects clause
  writes only. A regression that silently changes the log name or folds the
  empty list back into `all()` fails the test.

**Commit:** `fix(documents): report graph relationship writes truthfully (all([]) vacuity)`

### Phase C — bulletproof the pipeline (the north-star body)

This phase **is** the harden change's ingestion track. Do not write a second
spec; execute inside `openspec/changes/harden-agent-saul-evidence-pipeline/`
and tick its boxes. Entry order (verification before enforcement, matching the
change's own Phase 0 discipline):

| Step | Work | Where in the change | First file to touch |
|------|------|---------------------|---------------------|
| C1 | Two-contract PDF fixture: do parties/dates/governing-law leak across contracts? | task 1.8 | `evals/` or `tests/fixtures/` |
| C2 | Scanned-page OCR fixture: what coverage/confidence signals exist? | task 1.9 | `tests/fixtures/` |
| C3 | Upload guardrails: documented max size + extension allowlist enforced **before** full buffering | task 2.4 | documents router/DTO |
| C4 | Per-document cost/latency/model-call/Celery budgets | task 2.5 | worker + extraction service |
| C5 | Multi-contract classification boundary | Phase 4 | `classify_document` (regex) path |
| C6 | Carry `contract_index` through chunks → Graphiti → evidence | Phase 4 | chunk schema + writers |
| C7 | Fidelity gates: source coverage / order / table / section / OCR signals attached to document status | Phase 4 | `service/ingest.py`, status model |
| C8 | Untrusted-document prompt-injection fixtures | Phase 4 | `tests/` |
| C9 | Truthful terminal states + outbox/UI notifications, cache/idempotency scope keys, dead-letter handling | Phase 4 | status model, worker |
| C10 | Ingestion-specific gold set + CI gate | Phase 4 + Phase 8 | `evals/` |

**Acceptance:** each step has its verification outcome written in
`tasks.md` (the change's own convention). The phase is done when a contract
upload either completes fully within budget **or** returns a typed,
specific degradation — never a silent partial.

**Commit cadence:** one commit per step; the change's task 2.7 baseline rule
applies (stop rollout on any scope violation).

### Phase D — the graph bet (todo 136: relationships, ~2–4 sessions)

Now the `[]` from Phase B gets filled. The knowledge graph currently has
clause nodes and zero edges; for a legal corpus, edges are the product
("every obligation linked to Party X across agreements").

**Design constraints (carried from the Oct 10 analysis — still valid):**

- **Two-pass LangExtract, not `extraction_passes`.** `extraction_passes: 3`
  re-runs the *same* prompt for recall. Pass 2 here is a *different prompt
  over different input*: pass-1 entities + attributes fed back, output
  constrained to typed relationships. The TODO at
  `src/app/extraction/langextract/langextract_to_graph.py` (~line 82, verify
  on the post-restructure tree) already sketches this.
- **Grounding composition:** pass 2's `from_node`/`to_node` must resolve to
  pass-1 entity identities — never fresh strings. If they don't, Graphiti
  happily creates duplicate endpoints and you get edge soup. `LegalRelationship`
  already requires `clause_id` + `Citation`; that schema is telling you this.
- **Write through the existing idempotent Graphiti writer**
  (`write_relationship_edge` / the Phase-2 path of
  `write_clause_episodes_to_graphiti`) — no second writer, no raw
  `neo4j_client.merge_node`.

**Files:**

- `src/app/extraction/langextract/` — second-pass prompt + service call.
- `src/app/features/documents/service/ingest.py` — replace `[]` with the
  pass-2 output; delete the Phase-B skip log (or demote it to debug).
- `src/app/retrieval/graphiti/write_clause_episodes.py` — only if the typed
  relationship path needs a schema tweak.

**Acceptance:**

- Seeded two-party fixture yields the expected relationship edges with
  citations resolving to pass-1 `clause_id`s.
- Re-ingesting the same document writes zero duplicate edges (idempotency
  guard verified).
- `graph_relationships_skipped` no longer appears in logs for contract docs.

**Commit:** `feat(graph): typed relationship extraction via second LangExtract pass`

### Phase E — the quarantine ledger (implements "don't delete")

Create `docs/quarantine-ledger.md` — one table, reviewed whenever a phase
lands:

| Surface | Path | Kept because | Port-or-discard criterion | Revisit trigger |
|---------|------|--------------|---------------------------|-----------------|
| Ingestion HTTP service | `src/app/features/ingestion/` (unmounted) | A deliberate second HTTP surface is still a live product question | If the harden change's retrieval/ingestion consolidation (its Phase 5) ever needs it → port with the key-set contract test below; if not → discard | Harden change Phase 5 kickoff |
| Docling ingest_v2 | `src/app/extraction/docling/ingest_v2.py` | Hybrid chunker may beat policy chunking for some doc classes | Only if a golden-set comparison shows a win | After C10 gold set exists |
| LangExtract batch processor | `src/app/extraction/langextract/langextract_batch_processor.py`, `docling_preprocessor.py` | Batch mode may matter for corpus re-processing | Only if bulk re-ingestion of the existing corpus becomes a real task | First bulk re-index |
| Retrieval graph | `src/app/features/documents/retrieval_graph/` + `ask_via_retrieval_graph` | The "intended survivor" — unification target, not a duplicate to kill | Survives when `/search/ask` + `/legal/ask` route through it (hardening Phase 5) and the inline duplicate in `ask` is deleted | Hardening Phase 5 |

**One standing rule recorded in the ledger:** any future re-wiring of
`features/ingestion/service.py` must first add the boundary contract test —
`set(initial_state) <= set(DocumentIngestionState.__annotations__)` — because
`TypedDict(total=False)` + LangGraph's silent key-dropping make the type
system load-bearing nowhere on that edge. The old bug (path B's state keys
sent to path A's graph) died with `ingestion_kb`; this is the seatbelt for the
next re-wiring.

### Deferred — re-ingestion exposure (decided 2026-10-28)

Re-ingestion stays **unexposed** — not as a product feature, not as an admin
endpoint, and never as a model-callable tool — until all of the following
exist:

1. **Pipeline versioning** — `pipeline_version` persisted on chunks and
   surfaced in provenance (Phase C9 / harden change).
2. **Version-aware cache and idempotency keys** — search/answer caches and
   idempotency records must key on document + pipeline versions, or a re-run
   serves stale answers as fresh (C9 scope keys).
3. **Graphiti replacement semantics** — Phase D's idempotent writer decides
   what re-ingestion means for existing edges: replace, version, or supersede.
4. **Old-version retention** — a retention policy for superseded chunks, since
   open evidence handles and past citations may still point at them.
5. **Stable evidence identities** — rules for whether `chunk_id` survives a
   re-run; otherwise every historical EvidenceRef dangles.

When those land, re-ingestion enters as an **approval-gated administrative
action** (operator role, audited) — not a general endpoint and not an agent
capability. Until then the LangExtract batch processor stays quarantined per
the ledger above; its revisit trigger ("first bulk re-index") is the same
event that unlocks this item.

### Phase F — retrieval unification and agent tools (delegated)

Not specced here — the harden change owns both, and its specs are better than
anything this plan would duplicate:

- **Todo 252 (retrieval):** hardening change **Phase 5** ("Shared retrieval
  and evidence lifecycle"). The governing rule is already in the tree:
  `ask_via_retrieval_graph` is the intended survivor; the inline duplicate in
  `ask` is the thing that eventually gets deleted — *one implementation of
  plan→retrieve→grade→generate, never two*.
- **Todo 242 (agent tools):** hardening change **Phase 6**. Dependency note:
  graph-reading tools (`query_knowledge_graph`, `get_obligation_chain`) are
  only worth wiring **after Phase D** — tools over an edgeless graph return
  empty results that look like success, which teaches the agent to answer
  confidently from a graph with no edges. Tool best-practices when it lands:
  explicit Pydantic `args_schema` everywhere (kill the inferred-signature
  drift), read-only under authorized runtime context, writes approval-gated —
  all already specced in the change.

---

## 4. What changed since the Oct 10–11 analysis (for future sessions reading both documents)

- `ingestion_kb` (path B, ~1,000 lines incl. `nodes.py` 987 L) — **deleted** by
  `3bd67da fix(foundation): … delete dead trees`. `build_ingestion_graph` has
  zero references. The "delete vs harvest" question for B is moot.
- `test_unprovisioned_graph_fails_closed.py` — no longer present under that
  name; re-check `tests/` before assuming its claims.
- `shared/rag/` — fully dissolved into `src/app/extraction/` +
  `src/app/retrieval/`. Paths in the old analysis are stale; this file's
  Section 1 paths are current.
- Orphans D and E **survived** the restructure (now under `src/app/extraction/`).
- The `[]` at the relationship callsite, the miswired mount, the false D17
  docstring, and the unmounted retrieval graph — **all still live** as of
  2026-10-27 and addressed by Phases A–F above.
- New since the analysis: `openspec/changes/harden-agent-saul-evidence-pipeline/`
  (86 tasks, 0 done) — the umbrella this plan executes within.

---

## For the chosen ones

- **Connectivity is what refactors validate — nothing else.** The 503 endpoint
  has now survived a five-commit domain refactor *and* the dead-tree purge,
  because breakage requires a caller. When you audit a codebase, inventory
  callers, not correctness: `rg` for who invokes a thing is a stronger
  statement than any amount of reading it.
- **Quarantine converts a hard decision into a deferred one, and that's a
  feature — if it's written down.** "Don't delete for now" is only safe
  because the ledger names a port-or-discard criterion and a revisit trigger
  for every kept surface. A quarantine without a trigger is just a landfill
  with better PR.
- **The vacuity fix is load-bearing for the graph bet, not hygiene.** Phase B
  installs the signal (`graph_relationships_skipped`) whose disappearance is
  Phase D's success criterion. Fix the lie first, and the feature's rollout
  measures itself.
- **`all([]) is True` is the observability twin of "built but disconnected."**
  Both report health because nothing ran. Any gate computed with `all()` or
  `any()` over a possibly-empty collection needs a companion non-empty
  assertion when the empty case is semantically different — health checks,
  budget gates, write-result aggregation.
- **With `ingestion_kb` gone, the surviving state mismatch in
  `features/ingestion/service.py` is the last live instance of the
  LangGraph silent-drop class on this codebase** — `ainvoke` maps input onto
  declared channels and discards the rest, `total=False` TypedDicts make every
  channel optional, so a disjoint state dict type-checks clean. The only gate
  that catches the class is a key-set assertion at the boundary. That's why
  the ledger makes it a precondition, not a suggestion.
