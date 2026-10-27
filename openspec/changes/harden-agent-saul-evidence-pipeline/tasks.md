> Change class: **L**. Tasks are ordered by dependency and each item has a concrete verification outcome. No production implementation is included in this proposal.

## 1. Phase 0 — verify current behavior before changing it

- [ ] 1.1 Run the V1 two-user `search_legal_precedents` test with identical statute-like chunks, capture the exact SQL/predicate and record whether a cross-user leak occurs.
- [ ] 1.2 Trace every read and write of client-supplied `WSStartMessage.permissions` and `WSResumeMessage.reviewer_role` across DTO, service, graph, and HITL paths; attach the call-path inventory to the change notes.
- [ ] 1.3 Search the frontend repository and 14 days of access logs for `ingestion/documents/upload`; record zero non-4xx callers or name and preserve each consumer.
- [ ] 1.4 Confirm the current platform auth roles and absence/presence of tenant membership; record the result and identify the 1:1 backfill source.
- [ ] 1.5 Read and document state-hydration migrations, saved-state versions, and resume behavior; add a fixture for the oldest supported checkpoint.
- [ ] 1.6 Spike `ToolRuntime` access inside compiled sub-agents and prove a tool can read server-set context without model-visible authority; if it cannot, record the validated closure-based fallback.
- [ ] 1.7 Exercise Graphiti verification/write behavior and `/legal/ask` during Graphiti outage; record whether `TransientExternalError` escapes and the current `graphiti_verified` semantics.
- [ ] 1.8 (V8) Trace classification/legal-metadata scope and run a two-contract PDF fixture; record whether parties, dates, and governing law are mixed across contracts.
- [ ] 1.9 (V9) Inspect Docling OCR configuration and run a scanned-page fixture; document OCR on/off and available confidence/coverage signals.
- [ ] 1.10 (V10) Inventory document deletion behavior for originals, chunks, Graphiti episodes, caches, idempotency records, and read logs; compare current removal with the desired policy-gated behavior.

## 2. Phase 1 — ship independent security hotfixes

- [ ] 2.1 Replace risk/compliance placeholders with typed `not_assessed`/`not_assessable` states, reasons, and state-version bump; test that no path renders an unsupported result as compliant.
- [ ] 2.2 Remove or ignore client permission/reviewer fields at the DTO edge and derive authority from authenticated claims at start and resume; test role revocation during a pause.
- [ ] 2.3 Add tenant/user/allowed-document predicates and applicable jurisdiction filtering to the Postgres precedent leg; make the V1 two-user test pass.
- [ ] 2.4 Decide the approved maximum contract size from measured corpus data, document it as configuration, enforce it before full buffering/parsing, and reject disallowed extensions independent of MIME headers; add typed oversized/disallowed-file tests.
- [ ] 2.5 Add per-document extraction cost, latency, model-call/token, and Celery task budgets; test that a large accepted document completes within budget or returns a typed failure/warning rather than running unbounded.
- [ ] 2.6 Add a permanent two-tenant fixture with identical statute chunks and run it against every existing legal tool; fail CI on any unauthorized result, count, cache hit, or citation.
- [ ] 2.7 Run `uv run ruff check --fix src/`, `uv run ty check src/`, `uv run pytest`, and `ast-grep scan src/`; record the Phase 1 baseline and stop rollout on any scope violation.

## 3. Tenant, authorization, and state migration

- [ ] 3.1 Add tenants and membership persistence with `(tenant_id, user_id, legal_role)` and a transactional 1:1 backfill; verify counts, uniqueness, and audit user preservation.
- [ ] 3.2 In the same migration, replace document deduplication `(user_id, sha256)` with unique `(tenant_id, sha256)`; test teammates resolve one document while different tenants receive separate identities and confirm the old constraint is dropped.
- [ ] 3.3 Move every document/chunk read predicate to tenant scope plus explicit allowed document IDs while retaining `user_id` only for audit; run query-level isolation tests.
- [ ] 3.4 Define immutable `AuthorizationContext` with authenticated user, tenant, legal role, allowed document IDs, allowed operations, and thread ID; ensure serialization/model exposure cannot leak authority.
- [ ] 3.5 Wire context through service, graph factory, and `ToolRuntime` (or the validated closure fallback); assert tools reject/ignore model identity and scope arguments.
- [ ] 3.6 Implement analyst/reviewer/admin operation policy, self-review audit/report marking, orchestrator pre-dispatch checks, and human-review checks on every resume; test all roles and mid-pause revocation.
- [ ] 3.7 Replace single `doc_id` session scope with `allowed_doc_ids` and `primary_doc_id`; validate every ID belongs to the tenant and is `completed` or `completed_with_warnings`.
- [ ] 3.8 Add state-version upgrade logic for old checkpoints, including old single-document state and placeholder fields; test old-thread resume and typed failure for unupgradeable state.
- [ ] 3.9 Add chunk provenance (`pipeline_version`, `graph_indexed_at`, optional `extraction_version`, and conditional `contract_index`) and backfill existing rows as `legacy`; expose provenance in DTOs and distinguish it from graph verification.
- [ ] 3.10 Include tenant, document versions, and pipeline version in search/answer cache keys and `IdempotencyGuard` keys; test no cross-tenant or stale-version reuse over both cache windows.
- [ ] 3.11 Add typed document status mapping/constraint, stuck-age detection, dead-letter-to-failed transition, and safe failure reasons; test no dead-lettered item remains `received`.
- [ ] 3.12 Emit queued ingestion terminal events through the outbox and notify the UI for ready/failed statuses; test idempotency, correlation, and worker-vs-HTTP ownership.
- [ ] 3.13 Configure checkpoint retention/pruning (24-hour pause, 7-day GC), compact obsolete tool outputs, and use `adelete_thread()`; test expiry and distinguishable teardown outcomes.

## 4. Ingestion hardening before evidence work

- [ ] 4.1 (I1) If V8 confirms mixing, detect contract boundaries before classification/legal-metadata extraction, run metadata per contract, add `contract_index` to chunks, and carry it into Graphiti episodes, evidence, findings, and reports; otherwise retain the fixture and recorded one-contract result.
- [ ] 4.2 (I1) Add a two-contract fixture asserting no finding names a party/date/governing law without its contract identity; fail or downgrade when boundaries are insufficient.
- [ ] 4.3 (I2) Implement source-to-chunk coverage and order checks, implausible chunk-count detection, table/section loss warnings, and OCR confidence/coverage warnings using the V9-discovered signals.
- [ ] 4.4 (I2) Run scanned-PDF, table-heavy, and clause-numbered fixtures and assert typed `completed_with_warnings` reasons are visible before Saul analysis; include the source-fidelity result in coverage.
- [ ] 4.5 (I3) Add injected-document fixtures with instructions in clauses and forged party names; assert LangExtract/metadata output remains source-derived and Graphiti writes cannot widen tenant/document/contract scope.
- [ ] 4.6 (I4) Create and version the ingestion gold set covering scanned PDFs, tables, clause numbering, annexures, multi-contract files, and Phase-3-style clause-kind documents; label source-to-chunk coverage/order rather than retrieval relevance.
- [ ] 4.7 Run the queued pipeline against I4 and publish a machine-readable fidelity report; block the pilot if expected coverage/order or warning outcomes fail.

## 5. Shared retrieval and evidence lifecycle

- [ ] 5.1 Implement one `retrieve(ctx, query, filters, k, budget)` service over fused retrieval, source narrowing, optional Graphiti narrowing/coverage, reranking fallback, and token budgeting; preserve existing parity tests.
- [ ] 5.2 Add explicit Postgres, Redis, Graphiti, and reranker timeouts and classify `GraphitiError`/`TransientExternalError`; add a legal-path Graphiti outage test that preserves canonical chunks.
- [ ] 5.3 Centralize row hydration (`_row_to_chunk`) and return `page_no`, document version, pipeline version, contract index when applicable, and graph coverage in retrieval DTOs.
- [ ] 5.4 Define opaque run-scoped evidence handles bound to thread, tenant, document version, and run; store only handles/IDs in state and reject forged, cross-thread, stale, or cross-tenant handles.
- [ ] 5.5 Add `search_evidence` with `contracts`/`statutes`/`precedents`/`all`, allowed-doc, jurisdiction, and clause filters; add `read_evidence_by_id` with neighbor and character budgets; add `get_ingestion_state`.
- [ ] 5.6 Implement one result envelope with `ok`, `no_evidence`, `index_not_ready`, `ingestion_failed`, `unauthorized`, `transient_failure`, `timeout`, and `budget_exhausted`, plus coverage/retryable/provenance; test empty-result ambiguity.
- [ ] 5.7 Put existing knowledge graph, obligation chain, statute, and precedent tools behind runtime context and uniform schemas; convert failures to structured results and delete `hybrid_retrieve_precedents` only after parity.
- [ ] 5.8 Bind tools per node (risk, compliance, orchestrator) with an assertion-tested registry; keep Graphiti write tools unwired and decide/delete `detect_graph_conflicts` from experiment evidence.
- [ ] 5.9 (O1) Emit counters for Graphiti unavailability, rerank fallback, grader fallback, `completed_with_warnings`, stuck documents, dead-letter events, `index_not_ready`, and `no_evidence`; add dashboard panels and sustained-rise alerts.
- [ ] 5.10 Freeze the compiled retrieval graph shape, run the multi-hop A/B, and record quality/latency/cost evidence before retaining or deleting it.
- [ ] 5.11 Document and test the deliberate `/legal/ask` `graphiti_verified` difference versus Agent Saul's canonical-chunk path, or align both paths without making Graphiti legal truth.

## 6. Agent Saul graph, grounding, and finalization

- [ ] 6.1 Replace normalization/segmentation with deterministic persisted-chunk-to-segment adaptation using `chunk_id` as clause ID; prohibit concatenation and return `ingestion_incomplete` when structure is missing.
- [ ] 6.2 Remove full `document_text` and full clause text from checkpointed state; fetch text at read time through evidence handles and assert serialized-state size limits.
- [ ] 6.3 Make entity extraction and relationship mapping plan-conditional and prefer persisted LangExtract metadata; test they do not run for plans that do not need them.
- [ ] 6.4 Define typed `Finding`, `EvidenceRef`, claim ledger, and coverage models; preserve structured risk/compliance output and include document/pipeline/page/quote provenance plus `contract_index` when applicable.
- [ ] 6.5 Implement blocking grounding: current-run handle, scope, version, and quote-substring checks followed by entailment on survivors; reject/downgrade unsupported or contradicted findings.
- [ ] 6.6 Add tests for adjacent-clause, definition, exception, absent-span, forged-handle, non-indexed-but-canonical, multi-contract, and Graphiti-outage cases; ensure no path advances on `verified` alone.
- [ ] 6.7 Update human review rendering for unsupported/contradicted/not-assessed items; enforce current reviewer role on every resume and record override reason, identity, `self_reviewed`, and contract identity.
- [ ] 6.8 Update finalization input to approved plan, atomic findings, verification status, unresolved/contradicted claims, overrides, coverage, ingestion state, and contract provenance; disclose warnings/graph coverage without reconstructing a summary.
- [ ] 6.9 Rebuild deep research as a document-scoped adapter over `search_evidence`, fix argument mismatch, and add a plan-level execution test.
- [ ] 6.10 Gate `persist_memory` behind authorized approval and a feature flag; assert rejected/unapproved reports cannot be stored.
- [ ] 6.11 Add per-run model/retrieval/fan-out/token/wall-clock ledger; bound corrective retrieval and return `budget_exhausted` without further expansion.

## 7. Deletion, audit, provider, and cleanup gates

- [ ] 7.1 (D1) Using V10 and legal retention input, define deletion/recovery policy and the minimum non-content read-audit record; keep delete disabled until the policy gate is approved.
- [ ] 7.2 Implement tenant-scoped, idempotent deletion of original, chunks, Graphiti episodes, caches, and idempotency records, retaining policy-permitted audit records; test no derived artifact remains.
- [ ] 7.3 (L1) Obtain legal/security validation of Gemini terms, retention, training settings, and provider handling for extraction, embeddings, and generation; block real client data until signed off.
- [ ] 7.4 Audit imports and consumers of the Phase-3 graph, `/ingestion`, `token_counter`, `structural_navigator`, `errors`, and stale PageIndex/shared-RAG references.
- [ ] 7.5 (H1) Tag the Phase-3 harvest commit and copy `canonicalize.py` plus contextualization prompt to a short-lived branch before deletion.
- [ ] 7.6 (H1/E3–E5) Record port-or-discard decisions with evidence links: canonicalization likely ports only if E5 supports it; contextual preamble only if E3 justifies one LLM call/chunk; `bm25_force_merge` only if E4 gains; parallel fan-out only with contextualization; discard universal `chunk_kind="clause"`/empty-warning/never-verified semantics.
- [ ] 7.7 If `/ingestion` has consumers, return 410 for the documented window; otherwise remove route/lifespan wiring only after V3 and A4 evidence are attached.
- [ ] 7.8 Delete the obsolete `ingestion_kb` graph and tests only after I4 fidelity, H1 decisions, and canonical queued-ingestion parity are green.
- [ ] 7.9 Delete `hybrid_retrieve_precedents` and stale tool paths after shared retrieval parity and dependency scans pass.
- [ ] 7.10 Fix stale bundle docstring, `text = ""` TODO, PageIndex mentions, and old shared/RAG references; run source formatting/lint/type checks after each cleanup step.

## 8. Evaluation, CI, and red-team gates

- [ ] 8.1 Create a versioned 30–50 question answer gold set covering direct, multi-document, multi-hop, unanswerable, missing-statute, conflicts, near-misses, definitions/exceptions, injection, and tenant isolation; double-label and record SME adjudication.
- [ ] 8.2 Implement deterministic retrieval and entailment report schemas with golden-set version, judge model/prompt, per-query rows, aggregates, and commit ID; keep default tests offline.
- [ ] 8.3 Record a retrieval baseline before claiming improvement; prove the live harness reaches seeded chunk IDs and reports wiring failure when every query is empty.
- [ ] 8.4 (E7) Measure ingestion fidelity on I4, including source coverage/order, warning reasons, OCR/table outcomes, and multi-contract identity; report separately from retrieval relevance.
- [ ] 8.5 (E8) Compare per-document and per-contract metadata on the multi-contract fixture and record whether contract-local extraction improves correctness and cost.
- [ ] 8.6 Run experiments for inline ask vs search/read/verify, shared retrieve vs bounded graph, deterministic vs LLM preamble (E3), `bm25_force_merge` (E4), entity canonicalization (E5), and prompt-injection/cross-tenant red-team.
- [ ] 8.7 Add permanent CI tests for two-tenant isolation, other-user IDs, unauthorized doc IDs, injected scope widening, forged/absent evidence, role revocation, not-assessable pass rendering, near-miss citations, multi-contract identity, and ingestion injection fixtures.
- [ ] 8.8 Add metrics/assertions for evidence recall, atomic entailment, unsupported claims, correct abstention, citation validity, scope violations, tool calls, p95 latency, tokens, cost, and ingestion fidelity; fail builds on any unauthorized read or poisoning.

## 9. Rollout, observability, and rollback

- [ ] 9.1 Add structured correlation/audit events and metrics for authorization denials, handle issuance/reads, role changes, grounding outcomes, contract identity, coverage, Graphiti availability, ingestion terminal states, fidelity warnings, retries, and budgets; verify raw legal text/secrets are absent.
- [ ] 9.2 Roll out Phase 1 hotfixes independently, then Phase 2 tenant/state/dedup migration with dual-read/one-write and verified counts/checkpoint resumes.
- [ ] 9.3 Enable Phase 3 ingestion hardening only after V8/V9 decisions, I1/I2/I4 tests, and the documented size/budget gate; show warning reasons before Saul analysis.
- [ ] 9.4 Shadow-compare old/new ranked IDs and run seeded provider outages, forged handles, cache-scope, resume-revocation, OCR/fidelity, and multi-contract tests before enabling new Saul reads.
- [ ] 9.5 Enable grounding/reviewer/persistence only after approved-output tests and both gold-set baselines meet gates; retain internal demo mode with visible `not_assessed`.
- [ ] 9.6 Keep D1 deletion disabled and L1 real-data gate closed until legal/security decisions are recorded; synthetic/approved fixtures remain the only pilot data.
- [ ] 9.7 Exercise feature-flag rollback to read-only queued status, old checkpoint reader, and legacy retrieval adapter; retain non-destructive tenant/provenance/audit data.
- [ ] 9.8 Kill the pilot on any cross-scope read, forged-handle success, role-revocation bypass, false compliant pass, metadata poisoning, missing terminal event, sustained degradation alert, provider governance failure, or approved p95/cost budget breach; record incident and rollback decision.
- [ ] 9.9 Run final verification: `uv run ruff format src/ tests/`, `uv run ruff check src/ tests/`, `uv run ty check src/`, `uv run pytest`, `ast-grep scan src/`, and `openspec validate harden-agent-saul-evidence-pipeline --strict`.

