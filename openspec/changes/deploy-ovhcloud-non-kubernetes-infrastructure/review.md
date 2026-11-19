# Review: deploy-ovhcloud-non-kubernetes-infrastructure

**Class:** L

## Reviewer summary

This is a cross-cutting infrastructure, security-boundary, runtime-packaging and application-startup change. The proposal correctly treats the existing GCP tree and production Compose file as evidence sources rather than deployable truth. The design now closes the code-derived first-boot gaps: browser ownership, migration inputs, real health path, load-balancer probe semantics, feature-conditional secrets, explicit OTLP, ingestion model assets, resource/connection budgets, and multi-host crawler enforcement.

## Completeness

- [x] Every requirement has at least one `#### Scenario` with WHEN/THEN outcomes.
- [x] Edge and error cases include cross-environment destruction, missing quiet roles, provider-injected egress, metadata/private scans, unsupported service commands/extensions, browser/runtime drift, missing optional credentials, collector outage, active-task rollout, failed canary, restore failure and budget overrun.
- [x] Availability-sensitive attributes are quantified where measurement is meaningful: restore-verification age, alert-delivery time, budget variance, task shutdown margin, capacity profiles, retention, RPO and RTO.
- [x] No incomplete logical decision table is used. The three capacity profiles are alternatives with explicit SPOF claims rather than a condition matrix.
- [x] Every required runtime has an owner, health signal, resource/network expectation and recovery path.

## Correctness

- [x] Requirements use observable SHALL/MUST outcomes. Provider/tool choices remain in design and ADRs rather than masquerading as behavioral requirements.
- [x] `ovhcloud-infrastructure-deployment` is correctly ADDED. The three existing worker/crawler capabilities use ADDED requirements without replacing their current contracts.
- [x] The plan does not weaken Celery at-least-once, queue, retry, expiry, payload or Beat semantics.
- [x] The self-hosted PostgreSQL decision follows from the migration-required extension surface; the exact major remains evidence-gated rather than assumed.
- [x] The crawler claim matches the actual desired policy: host-enforced direct-egress denial plus a proxy that permits public destinations and rejects private/reserved/metadata targets.
- [x] `openspec validate deploy-ovhcloud-non-kubernetes-infrastructure --type change --strict` passes on 2026-11-16.

## Standards

- [x] Planned application changes stay in lifecycle/configuration/capability ownership; they do not add orchestration to routers or persistence logic to services.
- [x] Shared clients remain lifespan/app-state owned, but browser creation is removed from the API lifespan and retained in the crawler worker capability where it belongs.
- [x] All new I/O paths are required to remain async or external deployment jobs; dependency injection and public envelopes are unchanged.
- [x] Secret handling retains `SecretStr`/placeholder validation, makes requirements feature-aware, and keeps plaintext out of Terraform state, images, plans and ordinary outputs.
- [x] Apply tasks must run `uv run ruff format src/ tests/`, `uv run ruff check src/ tests/`, `uv run ty check src/`, `uv run pytest`, and `ast-grep scan src/`, plus infrastructure-specific validation.
- [x] The design is coherent with ADR-001 through ADR-009 and creates no repository-level ADR tree.

## Risk

- [x] Security: public-port bypass, crawler direct egress, provider egress defaults, metadata access, role credentials, secrets in state, mTLS rotation and private metrics are explicit.
- [x] Performance: Uvicorn/prefork multiplication, browser trees, ingestion models, database pools, OOM priorities, log retention and host headroom are explicit capacity inputs.
- [x] Data integrity: one-shot migrations, single Alembic head, tenant reconciliation, forward-only rollback, WAL/object backups and isolated restore drills are mandatory.
- [x] Availability: baseline SPOFs are named; HA is not claimed for one-node RabbitMQ/PostgreSQL/Neo4j or a single region.
- [x] Breaking impact: legacy deployment commands cease to be authoritative, but public HTTP/WebSocket/MCP/Celery contracts are unchanged.
- [x] Migration: test-first provisioning, exact-SKU gates, canary traffic, worker drain, fenced Beat, rollback pointer and legacy-GCP disposition are ordered.

## Contracts

- [x] Discovery states that no async interface is introduced or altered.
- [x] `contracts/asyncapi.yaml` contains the permitted one-line no-async note.
- [x] Existing task, queue, timer, expiry, result and WebSocket identities are preserved, so no subscriber migration is required.

## Advisory implementation conditions

- The first PostgreSQL task is a disposable executable compatibility proof, not production provisioning.
- The crawler negative tests must originate inside the final container and include raw IP/direct connection attempts, not only DNS/proxy tests.
- Production activation remains blocked until exact flavor/SKU cost, alert receiver, public DNS/certificate, and achieved restore evidence are recorded.
- `infra/gcp/` is neither mutated nor deleted until the active-use audit resolves its disposition.

## Independent-review remediation

The first DeepSeek verification returned `REVISE` at 0.58. Its evidence remains immutable in [Verify final OVH OpenSpec at 95% confidence](t3-thread://v2/61abb539-a90d-4f67-89b9-cbef6393413b/thread:delegated-task:command%3Amcp%3Aa3932251-ad09-4027-958e-180208a5c07f%3Adelegate-task%3A1c4c2776-a457-4a8e-bf4e-11ee88bce05c). The artifacts were corrected as follows:

1. Worker grace now derives from registered per-role maxima: default 300 seconds, ingestion 1,620 seconds and crawler 360 seconds, with at least 30 seconds in both Compose and systemd.
2. Early infrastructure tasks apply only test and render production plans; explicit later tasks apply, release, accept, restore and soak production after promotion approval.
3. Optional integration semantics are fixed by ADR-008: a typed explicit enablement set, never credential-presence inference.
4. Compose `stop_grace_period` and systemd `TimeoutStopSec` are both verified against the same task registry.
5. Production requires an isolated restore drill and a minimum 24-hour post-apply soak before acceptance.
6. Ingestion saturation, missing quiet subsystems, browser/library drift and final-image browser failure now have direct executable tasks.
7. Beat fencing is fixed by ADR-009's environment-scoped Valkey lease rather than unspecified ordering.
8. PostgreSQL/Valkey/RabbitMQ client budgets, pinned ingestion model revisions and exact Smokescreen TCP 80/443 policy are explicit.
9. The legacy-GCP scenario now matches the non-destructive design: record retain-versus-follow-up disposition rather than imply deletion in this change.
10. Production promotion now consumes a signed selected-region report proving required services/SKUs, measured latency, residency and quotas before apply.
11. Production apply regenerates a fresh post-promotion plan and binds it to signed region, profile, SKU, PostgreSQL digest/major and release-digest evidence; early preview plans cannot be applied and stale/mismatched/tampered evidence is rejected.
12. Missing-subsystem acceptance now removes each default, ingestion and crawler worker role as well as Beat, relay, compatibility and backup jobs.
13. Beat fencing now uses atomic owner-token acquire/renew/release semantics and forbids publication after lease renewal failure.
14. Evidence-binding rejection now executes before production apply and emits a plan-specific attestation that apply requires; negative stale/mismatch/tamper cases cannot run after mutation.

## Final independent confidence record

- **Result:** APPROVE
- **Confidence:** 0.99
- **Scope:** final task 12.3 evidence-order delta plus strict OpenSpec validation
- **Evidence:** [DeepSeek final evidence-order check](t3-thread://v2/61abb539-a90d-4f67-89b9-cbef6393413b/thread:delegated-task:command%3Amcp%3Aa3932251-ad09-4027-958e-180208a5c07f%3Adelegate-task%3Afa676c5e-ec83-4f38-ac41-8b9c8112de8b)
- **Provenance:** the task retained its DeepSeek title, but the completed run was resumed by the configured Copilot provider; it is recorded as an independent Copilot approval rather than mislabeled as a DeepSeek model result.

## VERDICT

VERDICT: APPROVED
