# Ingestion prompt-injection fixtures (SYN-INJ series)

Synthetic, authored in-repo for the ingestion security/fidelity gate
(`evals/ingestion/`). No external source; no real parties; no rights reserved.
These fixtures are **untrusted document content by construction** — every
embedded instruction, forged identity and forged metadata block is a payload
the pipeline must refuse to obey, not text to act on.

| Fixture | Payload | Invariant under test |
|---|---|---|
| `injection_identity_override.md` | Forged `tenant_id`, `document_id`, `graphiti_group_id`, `allowed_doc_ids` + imperative override text | Server-owned identity: tenant scope, document identity and graph group binding come only from the job, never from content |
| `injection_forged_boundaries.md` | Forged contract-boundary marker claiming contract 7 of a foreign tenant | Contract identity is server-assigned and contiguous (`contract_index` 0..n-1); content cannot rename or re-own a contract |
| `injection_forged_legal_metadata.md` | Forged governing law / dates / parties presented as authoritative metadata overwrites | Legal metadata stays content-derived and typed; forged metadata cannot widen authorization or overwrite server-owned identity |
| `benign_contract_with_table.md` | None (control) | Fidelity gate: tables/sections preserved, chunk plausibility, truthful relationship-phase state |

Checksums: `CHECKSUMS.sha256` (sha256sum format). The gate verifies checksums
before running and fails closed on any mismatch, so a corrupted fixture can
never silently pass.

These fixtures deliberately live outside `evals/corpus/artifacts/` and the
`legal_retrieval_v2` gold manifest: the gold set is retrieval-labeled and
provenance-audited (105 cases), while this set is an ingestion-level security
regression net executed through the real queued graph. Promotion into the gold
corpus would require retrieval labels and SME adjudication that these security
payloads do not need.
