> Change class: **L**. Role: pre-implementation reviewer.

## Completeness

- [x] Proposal, design, three new capability specs, and eight existing-capability delta specs are present.
- [x] Every added requirement has at least one concrete WHEN/THEN scenario; security, outage, migration, pause/resume,
  dead-letter, budget, prompt-injection/scope, and truthful-abstention edges are named.
- [x] The capability list covers the security boundary, canonical ingestion, retrieval, checkpoint state, tool binding,
  grounding, existing HITL handoffs, and evaluation.
- [x] Delta checks V8–V10 and ingestion-hardening items I1–I4 are traced into tasks, fidelity/contract specs, and the
  migration sequence before retrieval.
- [x] O1 degradation counters, H1 Phase-3 harvest decisions, E7/E8 experiments, D1 deletion/audit, and L1 provider
  governance are represented as explicit requirements or gated prerequisites.
- [x] The implementation order and exit gates are explicit: Phase 0 verification, S1–S5 hotfixes, tenant/state
  migration, retrieval/handles, grounding, experiments, then cleanup.

## Correctness

- [x] Requirements use normative MUST/SHALL language and describe observable behavior rather than Python classes or
  particular database statements.
- [x] Scenarios are testable and distinguish `no_evidence` from unavailable, unauthorized, incomplete, timeout, and
  budget states.
- [x] New capabilities contain Purpose sections of sufficient length. Existing capability deltas omit Purpose.
- [x] The deltas use ADDED for new security/provenance/coverage contracts and preserve the archived base requirements.
  If implementation materially replaces an existing base requirement rather than adding a constraint, the author must
  copy that full base block under MODIFIED before implementation; this is a task-level review check because current
  requirements remain valid and the new contracts tighten them.
- [x] Breaking public inputs (client permissions/reviewer role and single `doc_id`) are called out in the proposal,
  design migration, and compatibility plan.
- [x] S4 no longer assumes 10 MiB: the approved maximum is data-informed, enforced before buffering, and paired with
  per-document cost/latency/model/task budgets.
- [x] A1 changes deduplication in the same tenant migration from `(user_id, sha256)` to `(tenant_id, sha256)` and has
  teammate/different-tenant scenarios.
- [x] D1 deletion is policy-gated and L1 provider review blocks real client data; neither unresolved decision is hidden
  behind an implementation default.

## Standards

- [x] Design respects the modular layering: routers at the edge, services/repositories for domain work, worker-owned
  queue execution, and state dictionaries/typed failures for LangGraph nodes.
- [x] Authorization is injected as a narrow context rather than global mutable state; model-visible arguments cannot
  grant identity or scope.
- [x] The design requires typed feature errors/results at expected failure boundaries and explicit transport handling;
  it does not introduce `match`/`case` on `Success`/`Failure`, blanket catches, or secret logging.
- [x] Async timeouts, bounded fan-out, cache/idempotency scope, and provenance are explicit.

## Risk

- [x] Cross-tenant disclosure is treated as a release-blocking defect with permanent fixtures, forged-handle tests,
  fail-closed predicates, and a read kill switch.
- [x] State/schema migrations have dual-read/one-write sequencing, backfill verification, versioned checkpoint
  hydration, rollback before destructive cleanup, and a retention window.
- [x] False legal assurances are controlled by `not_assessed`, atomic findings, deterministic quote checks,
  entailment, role-gated review, coverage disclosure, and approval-gated persistence.
- [x] Graphiti outage is a coverage degradation rather than a source-of-truth failure; typed statuses prevent false
  negative answers.
- [x] Performance/cost risks have per-run budgets, explicit provider timeouts, p95/token/cost evaluation, and bounded
  corrective retrieval.
- [x] Ingestion risks have V8/V9 discovery, contract identity, source-fidelity/OCR/table/section warnings, injection
  fixtures, and a separate ingestion gold set rather than relying only on retrieval scores.
- [x] Destructive cleanup has H1 port-or-discard evidence requirements and D1 retention/audit gates.

## Delta Traceability Map

| Delta | Proposal | Design / ADR | Spec contract | Tasks / review gate |
|---|---|---|---|---|
| V8–V10 | Ingestion hardening, deletion/audit, provider-gate bullets | Phase 0 and open questions; contract/fidelity and deletion ADRs | Ingestion, grounded findings, deletion/audit specs | 1.8–1.10; discovery checks reviewed as preconditions |
| S4 | Data-informed limit and per-document budgets | Contract-aware ingestion decision; budget risk/migration | Evidence-security upload/budget requirement | 2.4–2.5, 9.3 |
| A1 | Tenant-aware dedup impact | Same Phase 2 tenant migration | Retrieval-schema dedup requirement | 3.1–3.2, 9.2 |
| I1–I4 | Multi-contract, fidelity, injection, ingestion gold-set bullets | Ingestion hardening phase and contract/fidelity ADR | Ingestion pipeline and grounded-findings requirements | 4.1–4.7, 6.4/6.6, 8.4–8.5 |
| O1 | Observability/degradation bullet | Retrieval phase and observability risks | Legal retrieval degradation counters requirement | 5.9, 9.1, 9.8 |
| H1; E3–E5 | Cleanup precondition and experiment changes | Phase 6 harvest/port-or-discard migration | Retrieval-evaluation experiment requirement | 7.5–7.8, 8.6 |
| D1 | New deletion/audit capability and pilot gate | Deletion/provider-data decision and open policy question | New `document-delete-and-access-audit` spec | 7.1–7.2, 9.6 |
| L1 | Provider review and no-real-data gate | Provider-data ADR and migration/rollback gate | Policy prerequisite, no model-tool behavior | 7.3, 9.6, 9.8 |
| Exit criteria/open items | Delta gates and unresolved decisions | Migration ordering and open questions | Covered by the above normative contracts | 9.2–9.9 |
- [x] Durable architectural choices (server context, canonical chunks, shared retrieval, tenant migration, grounding
  gate, queue status, and budgets) are recorded in design.md and should be mirrored in the ADR artifact.

## Verdict

**VERDICT:** `APPROVED`

The artifacts remain ready for granular task breakdown. Implementation is intentionally gated on the V8–V10 findings,
the data-informed S4 size decision, D1 legal retention/audit input, and L1 provider review. Preserve the existing Saul
transcript/HITL contract. Compare each affected base requirement and add a full MODIFIED block only if implementation
changes its prior observable behavior; the current delta adds constraints while preserving the base contracts.

## Follow-up audit

The pre-implementation audit compared the artifacts with the source plan's locked decisions and found four
traceability gaps. They are now closed in the artifacts: the grounded-findings contract defines the compliance
tri-state and required reason codes; the tool-registry/design/ADR contracts make native `@tool` bindings, HTTP-only
uploads, no model-callable resume/approval/re-ingestion, and read-only-versus-mutating approval gates explicit; the
legal retrieval spec records the deliberate `/legal/ask` `graphiti_verified` distinction; and tasks 1.8, 5.12, and
5.13 provide verification coverage for these boundaries and the transcript handoff invariant. No production code is
part of this change.
