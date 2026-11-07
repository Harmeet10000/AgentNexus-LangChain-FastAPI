# Phase 0 verification records (V1–V10)

Investigative records for tasks 1.1–1.10. Several verifications were
superseded by hardening work that landed during the change; each section
records current behaviour with code/test pointers.

## V1 — two-user `search_legal_precedents` leak (task 1.1)

Superseded by task 2.3 (the hotfix landed with its fixture). The two-user
fixture with identical statute chunks now exists at
`tests/unit/features/documents/test_precedent_tenant_isolation.py`:

- The Postgres statute leg (`_search_statutes_postgres` in
  `src/app/features/documents/application/tools/search_legal_precedents.py`)
  binds `tenant_id` and the explicit `allowed_doc_ids` on every statement —
  both ranked `text()` branches (pg_textsearch BM25 + pg_trgm) and the fused
  detail `Select`. An empty allowlist fails closed and issues no corpus read.
- The Graphiti leg is scoped through `_scoped_group_ids` and refuses an empty
  allowlist (`src/app/retrieval/graphiti/client.py`).
- The historical V1 defect (user-scoped/unscoped statute SQL) cannot recur:
  the fixture asserts the other tenant's tenant id and document ids never
  appear in the caller's binds or result.

Jurisdiction note: the Postgres corpus carries no per-chunk jurisdiction —
`documents.jurisdiction` exists as a document-level column, but the precedent
tool records `jurisdiction: None` for statute rows (the corpus is
jurisdiction-agnostic; both precedent tools carry `# noqa: ARG001` on the
jurisdiction parameter recording this). Applicable-jurisdiction filtering on
this leg is therefore a recorded no-op; the filter is honoured on the Graphiti
leg and by the evidence contract (`search_evidence(jurisdiction=...)`).

## V2 — client authority trace (task 1.2)

Call-path inventory of the client-supplied authority fields:

| Field | Declared | Read | Verdict |
| --- | --- | --- | --- |
| `WSStartMessage.permissions` | `agent_saul/api/dto.py:43` | never read — `_build_initial_state` writes `"permissions": {}` (`application/service.py:1897-1899`, comment: "client permissions are never copied into graph state") | dead at the edge |
| `CreateSessionRequest.permissions` | `agent_saul/api/dto.py:196` | never read by the HTTP session router | dead at the edge |
| `WSResumeMessage.reviewer_role` | `agent_saul/api/dto.py:78` (deprecated compat) | ignored by `_apply_resume`; reviewer role re-derived from LIVE tenant membership (`TenantMembershipService.get_live_authorization`, `application/service.py:1566-1571, 1672-1698`); refusals `REVIEWER_ROLE_REQUIRED` (service.py:1463) and `AUTHORIZATION_CHANGED` on `role_revision` mismatch (service.py:1416-1426) | dead at the edge |
| `human_response["reviewer_role"]` read in `graph/nodes.py:2849` | — | consumes the resume payload the SERVICE built (server-derived), never the client frame | server-derived |

Graph-state residue: `LegalAgentState.permissions` (`domain/state.py:884`,
default `{}` at state.py:694) remains as a checkpoint-compat channel; legacy
hydration fills `{}` (`tests/unit/test_state_hydration.py`). The DTO fields
still exist on the wire for rolling-client compatibility — removing them is a
breaking client contract deferred to DTO cleanup, but no authority flows
through them (pinned by
`tests/unit/features/agent_saul/application/test_resume_authorization.py`).

Server-derived authority itself landed with 3.4: immutable
`AuthorizationContext` (commit `accf5a2`) and the WebSocket twin (commit
`c230052`, PR #112). What remains open is the wider role-policy wiring of
tasks 3.5/3.6 — not a client-authority gap.

## V3 — `/ingestion/documents/upload` consumers (task 1.3)

The route survives as the legacy shim `POST /ingestion/documents/upload`
(`src/app/features/documents/api/http/legacy_ingest_router.py`) with its
sunset gate (`API_SUNSET_DATE`), a `legacy_ingestion_upload_deprecated`
structured event naming the endpoint, and the same bounded upload validation
as the canonical route (`read_bounded_upload`). Surface enforcement lives in
`tests/unit/features/documents/api/test_legacy_upload_deprecation.py`; the
re-ingestion-readiness tests (`tests/unit/documents/test_re_ingestion_readiness.py`)
pin route/registry identity. No frontend repository exists in this monorepo
and access logs are outside the unit band, so the recorded answer is: the
legacy shim is the one known consumer surface, preserved behind its sunset
gate; removal/410 behaviour is task 7.7's decision.

## V4 — platform auth roles and tenant membership (task 1.4)

Landed with 3.1: `PersonalTenantProvisioner` (commit `a3d3296`, PR #109)
provisions a personal tenant + `(tenant_id, user_id, legal_role)` membership
in one `ON CONFLICT` transaction for every account-creation path. The 1:1
backfill source is the existing users table: each pre-existing user maps to
exactly one personal tenant (`personal:<user>` via `personal_tenant_id`,
stability asserted in
`tests/unit/documents/test_tenant_scope.py::test_personal_tenants_are_stable_and_distinct`)
with the transactional invariant tested in
`tests/unit/features/auth/test_provisioning_session_invariant.py` and
`tests/integration/test_auth.py`.

## V5 — state-hydration migrations (task 1.5)

Documented and fixed by `tests/unit/test_state_hydration.py`:

- The oldest supported checkpoint fixture exists: version 0
  (`test_legacy_version_zero_is_upgraded_with_missing_keys`), plus the
  absent-version shape and explicitly migratable versions 0–7
  (`_MIGRATABLE_STATE_SCHEMA_VERSIONS`, `domain/state.py:629`).
- Unknown/invalid versions raise typed `StateSchemaVersionError`; the session
  service refuses with `STATE_SCHEMA_INCOMPATIBLE` before any node runs
  (`_hydrate_or_refuse`, `_apply_resume` in `application/service.py`).
- `STATE_SCHEMA_VERSION` is 8; the v8 `checkpoint_id` channel is migrated to
  the evidence scope (test `test_v8_checkpoint_id_channel_is_migrated_to_evidence_scope`).

## V6 — `ToolRuntime` inside compiled sub-agents (task 1.6)

Proven by construction — no closure fallback is needed:

- The graph factory builds each specialist agent with
  `context_schema=ToolRuntimeContext` (`agent_saul/graph/factory.py:252,261`),
  so `ToolRuntime` reaches nested `create_agent` subgraphs.
- `trusted_tool_context`
  (`documents/application/tools/authorization.py`) takes identity, tenant,
  allowlist, operations, and capabilities from `ToolRuntime.context` only;
  `runtime.state` (model-controlled specialist state) is never a fallback for
  grants. `InjectedToolArg` keeps runtime fields out of the model-facing
  schema (`test_tenant_scope.py::test_evidence_tools_hide_identity_and_document_scope_from_model_schema`).
- Role→capability projections are asserted per role in
  `tests/unit/features/agent_saul/graph/test_policy_consistency.py`.

## V7 — Graphiti outage semantics (task 1.7)

- `/legal/ask` still sets `require_graphiti_verified=True`
  (`documents/api/http/router.py:209`) — the legacy ask path treats
  graph verification as a requirement, while the evidence/grounding contract
  treats it as a coverage signal (`mechanically_verified` default,
  `graph_indexed_only` policy). The difference is parametrized in the parity
  suite (`tests/unit/features/documents/retrieval/test_ask_inline_graph_parity.py`,
  `require_graphiti_verified` cases); aligning/documenting it fully is 5.11.
- Outage semantics: a `GraphitiAvailability.UNAVAILABLE` leg is reported as
  `unavailable_layers`/`basis_unknown` (precedent search) or a typed
  `unavailable_result` envelope (knowledge-graph tool) — never as fabricated
  "no results" (`tests/unit/test_tool_unavailability.py`). Canonical chunk
  retrieval stays available; coverage marks
  `graph_unavailable_documents` (evidence service).
- `TransientExternalError` is caught inside the retrieval graph
  (`src/app/retrieval/graph/nodes.py:209,432`) and mapped to typed degraded
  responses — it does not escape `/legal/ask` as an unhandled error.
- `graphiti_verified` itself is a persisted chunk column (migration `0003`),
  backfilled to `graph_indexed_at` provenance (migration `0031`), and exposed
  as `EvidenceRef.graph_indexed` plus the coverage counters — never as legal
  truth.

## V8 — two-contract classification/metadata scope (task 1.8)

Superseded by 4.1: `establish_contract_boundaries`
(`ingestion/chunking/classification.py:145`) detects contract boundaries
before metadata extraction, assigns `contract_index` per chunk, and carries
it into edge records and evidence (commits `397b8e8`, `360f053`, `221b67b`).
Parties/dates/governing law are extracted per boundary, so a two-contract PDF
no longer mixes metadata across contracts. The residual dedicated two-contract
fixture is task 4.2 (not Phase 0/1).

## V9 — Docling OCR configuration and signals (task 1.9)

Superseded by 4.3: source-to-chunk coverage/order checks, implausible
chunk-count detection, and table-preservation signals are implemented and
tested (`tests/unit/features/documents/test_ingestion_truth_fidelity.py`,
coverage/order/table tests from line 698; per-boundary plausibility commit
`49e752e`). OCR confidence/coverage warnings use persisted, versioned
thresholds with explicitly-unavailable states, folding into
`completed_with_warnings` (commit `b081a80`). The extraction-side resource
budgets are configuration (`DOCUMENT_EXTRACTION_TIMEOUT_SECONDS`,
`_MAX_PAGES`, `_MAX_OUTPUT_BYTES` in `src/app/config/settings.py`).

## V10 — document deletion inventory (task 1.10)

No delete operation exists today:
`application/commands/delete_document.py` is an explicit reserved placeholder
("no delete operation exists today"). Inventory of everything deletion WILL
touch once enabled (the input for tasks 7.1/7.2):

- Original file: `documents.source_uri`/`object_uri` on S3/R2 storage.
- Derived rows: `chunks` (canonical chunks, provenance columns,
  `graphiti_episode_id`, `graphiti_verified`).
- Graphiti: clause-episode edges/nodes within the document's group partition.
- Caches: search/answer caches and `IdempotencyGuard` records keyed by
  tool/step/document scope.
- Audit: the read/access audit records (`application/access_audit.py`) and
  the chat transcript projections — the minimum non-content read-audit record
  is what the D1 policy must decide to retain.
- Outbox: document lifecycle events already emitted for the document.

Deletion stays disabled for pilot data until the D1 policy gate (legal
retention input) closes — Phase 7, not Phase 0/1.
