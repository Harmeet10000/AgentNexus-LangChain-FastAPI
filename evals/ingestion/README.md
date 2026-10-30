# Queued-ingestion security and fidelity gate

This directory is the CI quality gate for the bulletproof-ingestion roadmap
(issue #107, plan sections "the pipeline never lies", "OCR/source fidelity",
"prompt-injection fixtures", "queued-ingestion CI quality gate").  It runs the
fixtures in `fixtures/` through the **real queued ingestion path** — the same
compiled `build_document_ingestion_graph` graph the Celery worker task
invokes — with hermetic fakes (in-memory object store, no-embedding provider,
no-model extraction, no database, no Neo4j, no network model calls).

Run it with:

```bash
uv run pytest evals/ingestion -q
```

CI runs exactly this command (see `.github/workflows/test.yml`,
"Enforce queued-ingestion fidelity and injection gate"); any assertion failure
fails the build.

## What is enforced (fails the build)

| Invariant | Proof |
|---|---|
| Fixture integrity | sha256 checksums in `fixtures/CHECKSUMS.sha256`; any mismatch fails closed before any run |
| Tenant scope | persisted chunk rows and document `ingestion_state` carry the server-issued tenant; the forged tenant in `injection_identity_override.md` / `injection_forged_boundaries.md` never appears as an identity field |
| Graph group binding | no `group_id`/`graphiti_group_id`/`allowed_doc_ids`/`tenant_id` keys can be planted by document content into chunk metadata |
| Contract identity | `contract_index` is a server-assigned subset of the conservative boundary function's contiguous range; the forged "contract 7 of tenant X" claim is inert |
| Table preservation | the control fixture's parsed table must survive into chunks (`table_loss_ratio == 0.0`) |
| Chunk plausibility | zero chunks for a non-empty document is `implausible` and degrades the terminal state to `completed_with_warnings` (fault-injected untruthful-success test) |
| Relationship-phase truth | the queued path persists `graph_relationships.status = skipped_not_implemented` (reason `relationship_extraction_not_implemented`) — never a vacuous `all([])` success |
| No fabricated numbers | OCR confidence/coverage are persisted as explicit `unavailable` when the hermetic path cannot measure them |

## Report-only (baseline first, enforcement after variance)

Source coverage and OCR confidence thresholds default to disabled
(`FidelityThresholds.source_coverage_min = 0.0`,
`ocr_confidence_min = 0.0`) per the plan's "CI first reports a baseline;
enforcement thresholds are enabled only after variance is understood".  The
metrics are still measured and persisted on every document.  Tightening them is
a one-line, versioned change in
`src/app/features/documents/domain/fidelity.py`.

## Deliberate scope boundary

These fixtures live outside `evals/corpus/` and the `legal_retrieval_v2` gold
manifest: the gold set is retrieval-labeled and SME-adjudicated (105 cases),
while this set is an ingestion-level security regression net that does not need
retrieval labels.  See `fixtures/README.md`.
