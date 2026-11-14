# Phase 9 observability audit (task 9.1)

Source contract: design.md §7 — "Logs carry correlation IDs, tenant/document hashes or IDs
permitted by policy, pipeline/state versions, result status, retryability, and latency; they never
carry raw legal text or secrets. Evidence handle issuance/read, authorization denials, role changes,
grounding outcomes, graph availability, and coverage are metrics/audit events."

Each row: the item named in task 9.1 → where it is emitted today (file + event/counter name) →
status. Filled gaps follow the `degradation.py` / `logger.bind` conventions; the no-secrets
verification lives in `tests/unit/observability/test_structured_events_no_secrets.py`.

## Audit table

| # | Named item | Emission point (file + event/counter) | Status |
|---|------------|----------------------------------------|--------|
| 1a | Authorization denials — document/evidence reads | `application/access_audit.py::DocumentAccessAuditEvent` (operation, outcome, reason, tenant/principal/thread/run/checkpoint correlation; `extra="forbid"`, no content field) appended by `application/evidence/service.py::_record_access` (`evidence.search`, `evidence.read`, `document.read`) and `application/queries/documents.py::_record_access` (`outcome="denied"` for an explicitly empty scope); audit failure is fail-closed (`document_access_audit_failed`) | EXISTS |
| 1b | Authorization denials — auth/session | `auth/application/service.py::_emit_security_event` → structured `auth_security_event` log + durable outbox event (`aggregate_type="auth_security"`; e.g. `auth.refresh_token_reuse_detected`), plus OAuth link rejection/replay events | EXISTS |
| 1c | Authorization denials — Saul WS/session | `application/service.py` refusal events with typed codes: `saul_t0_refused_run`, `saul_thread_refused` (code bind), `saul_resume_after_expiry_refused`, `saul_resume_document_readiness_refused`, `saul_first_frame_identity_refused`, `saul_legacy_plan_refused`, `saul_paused_turn_expired_resume_refused`, `saul_thread_paused_refusing_new_start`; `graph/nodes.py` `orchestrator_invalid_target` (operation policy) | EXISTS |
| 2 | Handle issuance/reads | Issuance correlates through the `evidence.search` access-audit row (run/thread/checkpoint-scoped handles minted inside the audited call); every read names the exact handle in the `evidence.read` audit row (`evidence_ids`, `document_ids`, stale/expiry in `outcome`/`reason`) | EXISTS (access audit; no separate counter — the audit row IS the issuance/read record) |
| 3 | Role changes | `auth/adapters/persistence/postgres/tenancy_repository.py::update_roles` — the only structured event is the failure path (`tenant_membership_update_failed`); a SUCCESSFUL role change (grant/revoke of reviewer/admin) emits no audit event | **GAP** — the emission point is in the auth feature, outside this change's file scope (`src/app/observability/**`, `src/app/features/documents/**`, `tests/**`, specs); recorded here as the one known gap, owned by the auth feature. Downstream effects of a role change ARE observable: `role_revision` bumps refuse mid-pause resumes (`AUTHORIZATION_CHANGED`, `tests/.../test_resume_authorization.py`) |
| 4 | Grounding outcomes | `agent_saul/graph/nodes.py` grounding node: `grounding_verified` (verified, status, validation_decision, validation_reason), `grounding_output_invalid`, `grounding_verifier_unavailable`; typed state errors `GROUNDING_GATE_REJECTED` / `GROUNDING_NOT_VERIFIED`; escalation through the HITL sink; `human_review_rejected` | EXISTS |
| 5 | Contract identity | Structured provenance, not a counter: `contract_index` on the chunk column, `EvidenceRef`, `EvidenceHandle`, and findings; typed boundary failures (`MULTI_CONTRACT_BOUNDARY_AMBIGUOUS`, `LEGAL_METADATA_VALIDATION_FAILED`); worker graph-phase events (`clause_episodes_written`, `graph_relationships_failed/skipped/written`) bound to `document_id` with the contract carried in `GraphIngestionContext` | EXISTS |
| 6 | Coverage | Counters `documents.evidence.no_evidence_total`, `documents.evidence.index_not_ready_total`, `documents.ingestion.completed_with_warnings_total` (`observability/degradation.py`); per-response `EvidenceCoverage` envelope (ask, evidence search/read, `get_ingestion_state`) derived from hydrated rows, never asserted | EXISTS |
| 7 | Graphiti availability | Counter `documents.graphiti.unavailable_total` + sustained-rise `document_degradation_alert`; events `graph_retrieval_failed`, `graph_relationships_failed`; outage degrades to canonical-only fusion in the service, the ask hybrid node, and the three legal tools | EXISTS |
| 8 | Ingestion terminal states | Outbox terminal events staged transactionally (`application/commands/upload_document.py::with_terminal_outbox`; `ingestion/workflow.py::_record_ingestion_failure`; relay dead-letter bridge `events/relay.py`) consumed by `ingest_document_terminal` → `documents_ingest_terminal`; `documents_ingest_queued`, `documents_ingest_stuck_detected`, `canonical_ingestion_completed`, `outbox_published`; counter `documents.ingestion.stuck_documents_total`; `documents.ingestion.dead_letter_events_total` defined-but-not-incremented (recorded 5.9 deviation) | EXISTS (with the recorded 5.9 deviation) |
| 9 | Fidelity warnings | Counter `documents.ingestion.completed_with_warnings_total`; typed `FIDELITY_*` reasons persisted into chunk warnings and surfaced in `DocumentStatusResponse.warnings` before Saul analysis (4.4 suite `test_fidelity_warning_visibility.py`); `document_extraction_failed` worker log with typed code (3.11); `decode_fidelity_metadata` decode-failure log | EXISTS |
| 10 | Retries | `resilience/retry.py` → `retry_scheduled` (attempt, waited, outcome); `ai/graph/retry.py` → `kb_retry_transient_retry`; `auth/application/revocation/pending.py` → `auth_pending_revocation_retry_scheduled`; outbox relay attempt-scoped redelivery with dead-letter bridge; Celery task time limits (`soft_time_limit=1500/time_limit=1620`) | EXISTS |
| 11 | Budgets | `retrieval/graph/budgets.py` → `stage_budget_degradation` + sustained `retrieval_budget_degradation_alert` (per stage + document class); extraction `ExtractionBudget` ledger → typed `BUDGET_EXHAUSTED` with `budget_dimension` riding `document_extraction_failed` + `document_extraction_budget_checkpoint_invalid`; Saul `BudgetLedger` exhaustion → typed `BUDGET_EXHAUSTED` errors in checkpointed state/transcript; evidence envelope `budget_exhausted` status | EXISTS |
| 12 | Raw legal text/secrets absent | `DocumentAccessAuditEvent` is `extra="forbid"` with no content/text/snippet/key field ("raw source text is unrepresentable"); `degradation.py` documents tenant-safe correlation only (hashes/IDs, "never legal text"); `budgets.py` binds stage/kind/observed/limit/document_class; retry binds carry attempt/waited/outcome only. Verified by test: `tests/unit/observability/test_structured_events_no_secrets.py` (planted legal-text/API-key markers never appear in any emitted event; bind keys restricted to the safe vocabulary; the audit model rejects content fields) | EXISTS (test added by 9.1) |

## Filled gaps

None in code. The single gap found (row 3, successful role-change events) cannot be filled within
this change's file scope — the emission point is `src/app/features/auth/`, which the concurrent
rollout owns; it is recorded here rather than patched out of scope.

## Verdict

Task 9.1 is satisfied by the audit above plus the no-secrets verification test. Every named category
except successful role-change events already has a structured, correlation-safe emission point.
