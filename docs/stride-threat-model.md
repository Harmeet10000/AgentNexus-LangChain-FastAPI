"""STRIDE threat model — Agent Saul (legal-corpus document intelligence).

Derived from the STRIDE-per-boundary walk over the shipped data flows, 2026-11-14.
Companion to the harden change's red-team suites (Phase 8) and the observability
program (Phase 5.9/9.1). Each finding cites its enforcement point and test.

## Trust boundaries walked

B1 user → WS/API (auth, session start/resume)
B2 API/worker → ingestion pipeline (Docling, extraction LLM, embeddings)
B3 Saul graph → tools (evidence seam, knowledge tools)
B4 tools → stores (Postgres pgvector/bm25, Neo4j Aura, R2, Redis)
B5 model outputs → persisted state (extraction, findings, verdicts)
B6 deployment boundary (this machine → cloud free-tier services)

## The six categories, mapped

### S — Spoofing
- Forged evidence handles: server-issued, run-scoped, hash-verified at read;
  the grounding gate rejects forged handles (test_grounding_gate_matrix).
- Forged party names / injected authority: injection fixtures prove extraction
  stays source-derived (evals/ingestion, commit c1ca96e).
- Client-supplied reviewer roles: DTO-stripped; authority re-derived from live
  membership on every resume (test_resume_authorization.py).
- Residual: S-1 — the WS session token lifetime vs long pauses (reviewed in
  Phase 9 rollout; accepted for synthetic-only pilot data).

### T — Tampering
- Metadata-override injection ("SYSTEM NOTE ... OVERWRITE METADATA"): the
  extraction prompt delimits untrusted content; persisted metadata stays
  source-derived (test_forged_metadata_cannot_displace_server_side_extraction).
- Delimiter forgery inside documents: delimiter neutralization (commit 49e752e).
- Boundary heading forgery: headings restricted to # / ALL-CAPS markers; the
  length arm that let ordinary clauses re-own boundaries was removed
  (red-team finding, fixed 2026-11-11 — see notes/phase8-eval-ci-redteam.md).
- Residual: T-1 — relationship-edge tampering inside Neo4j (the graph store is
  shared; writes are partition-scoped by group id — accepted for pilot).

### R — Repudiation
- Read audit: append-only DocumentAccessAuditEvent per evidence search/read,
  non-content fields only (test_structured_events_no_secrets.py).
- Deletion repudiation: tombstone + audit retention (90 days, APPROVED
  2026-11-14) — the tombstone precedes any load.
- Residual: R-1 — role-change events are not yet emitted (9.1 audit GAP;
  owned by the auth feature).

### I — Information Disclosure
- Cross-tenant reads: two-tenant isolation suites over all six legal tools +
  all 13 repository read predicates (query-level); dedup is tenant-scoped
  (migration 0029).
- Scope widening: model-supplied scope never reaches the graph (V6 tests);
  the key-set seatbelt gates every compiled graph's input channels.
- Grounding disclosure: unsupported/contradicted findings render in review
  with contract identity (6.7).
- Residual: I-1 — `_legal/ask`'s `graphiti_verified` policy discloses
  graph coverage counts (deliberate, documented — 5.11).

### D — Denial of Service
- Upload size/extension caps before buffering (2.4); per-document extraction
  budgets (2.5); stage budgets on every graph node (G6) with the standing
  degradation alert (5.9).
- **Observed non-malicious DoS (both live, 2026-11-11/13):**
  - D-1: a transient Timescale LISTEN timeout shut the whole API down —
    a dependency blip became a full outage. Fix owed: the listener loop must
    reconnect-and-degrade, not terminate serving.
  - D-2: the OOM killer terminated workers mid-ingestion (Docling's per-call
    model load ≈ 3.7 GB RSS on the 37 MB document). Fix owed: the singleton-
    converter follow-up in the parser (named in its docstring) + worker memory
    bounds are already configured (`--max-memory-per-child`).
  - D-3: the Gemini free-tier daily embedding quota exhausted mid-burst,
    failing 12/15 ingestions with no typed quota signal surfaced.
    **FIXED 2026-11-15** — quota-aware guard at the one embedding choke point
    (`app.ai.embeddings._quota_guarded`: seconds-scale probes for transient
    429s; a 429 that names the daily cap, or one that survives the probes,
    raises the typed `EmbeddingQuotaExhaustedError`). The documents twin parks the
    document in the NON-terminal `quota_exhausted` status with typed metadata
    (resumable by the next quota window's idempotent driver — never a
    dead-letter, never a sleeping worker); the search twin dead-letters
    immediately with the typed code (`SearchIngestQuotaExhaustedError`).
- Residual: D-4 — no rate limit on evidence-handle reads beyond auth (pilot-
  scale accepted).

### E — Elevation of Privilege
- Model-supplied scope arguments: never read — runtime context only (V6
  tests; test_model_state_cannot_widen_the_tenant_boundary).
- Key-set seatbelt: every compiled graph's input channels are contract-tested
  (`set(initial_state) <= set(State.__annotations__)`) — the LangGraph
  silent-drop class is closed.
- Operation policy: analyst/reviewer/admin operation sets, orchestrator
  pre-dispatch checks, reviewer-role checks on every resume (3.6).
- Residual: E-1 — role-change emission (the 9.1 GAP) would close the
  elevation-audit loop.

## Standing residuals (accepted, with owners)

| ID | Threat | Category | Owner / trigger |
|---|---|---|---|
| S-1 | WS token lifetime vs long pauses | S | Phase 9 pilot review |
| T-1 | Shared Neo4j write surface | T | harden 7.x / staging |
| R-1 | Role-change event emission | R | auth feature (9.1 audit) |
| D-1 | LISTEN timeout → full API shutdown | D | root-caused 2026-11-14: systemd-oomd SIGTERM of the cgroup under memory pressure (not a listener bug — the relay's reconnect loop is sound). Mitigated by D-2's memory fix; the relay degrades via its backoff loop |
| D-2 | OOM under Docling model load | D | **FIXED 2026-11-14** — process-wide singleton converter, convert serialised by lock (tests: test_parser_converter_singleton.py); trade-off documented in parse_document |
| D-3 | Embedding quota exhaustion untyped | D | **FIXED 2026-11-15** — typed `EmbeddingQuotaExhaustedError` at the embedding choke point (probe-then-classify: seconds-scale probes separate a per-minute rate limit from the daily cap); documents park in non-terminal `quota_exhausted` with typed metadata, search dead-letters typed (`SearchIngestQuotaExhaustedError`). Measured incident: 2026-11-14, free-tier 1000/day cap, 12/15 ingestions |
| D-4 | No secondary rate limit on handle reads | D | pilot-scale accepted |
| E-1 | Role-change audit loop closure | E | auth feature (with R-1) |

## Method note

This model was produced by the STRIDE-per-boundary walk over the shipped
flows, cross-checked against the red-team suites (Phase 8: 88/0 after the
boundary regression fix) and the live incidents observed during the
measurement runs. It is a living document: update on every new trust
boundary or dependency.
