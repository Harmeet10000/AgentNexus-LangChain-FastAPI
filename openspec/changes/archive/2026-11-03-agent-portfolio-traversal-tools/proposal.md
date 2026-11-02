> Change class (pick one): **S** single-file fix / config / bump / docs · **M** feature in one module · **L** cross-cutting (multi-module, migration, security, public API)

**M** — new tools inside Agent Saul's tooling over the existing Graphiti seam; no HTTP surface, no migration, no new dependency. **Gated change:** implementation is blocked on relationship edges flowing in the knowledge graph (retired plan Phase D / harden change Phase 4 relationship work) — this change is authored now so the gate, the contracts, and the cap discipline are agreed before the unblocking work lands.

## Why

Mission 2 of the tools plan — cross-document portfolio interrogation ("across all 50 agreements, which indemnification clauses conflict with our playbook?") — is impossible for any chat-only tool: it requires traversal over typed relationships between clause entities. Today the knowledge graph has clause nodes and zero edges, and Saul's tools are all point lookups. `PLAN-agent-tools.md` P2 names traversal as the asymmetric-impact payload: it changes what the agent can *do*, not just how fast it points at text. Authoring the contract now is what makes the cap discipline (swap, don't add) enforceable when the gate opens.

## What Changes

- **Three thin traversal tools** over the Graphiti relationship backend, each returning evidence handles + a bounded result envelope (never raw payloads):
  - `find_conflicting_clauses` — clauses whose typed relationships or attributes conflict across documents in scope;
  - `cross_document_obligations` — obligations of a party across all in-scope agreements;
  - `party_obligation_map` — the party→obligation web for a named party or set.
- **Fusion with the evidence seam.** Every traversal result resolves to evidence handles served by the existing `read_evidence_by_id` contract; traversal finds *where*, the evidence seam serves *what* — one read path, handle-scoped per run (D5's run-scoping applies unchanged).
- **Scope discipline.** All three tools take only model-visible query arguments (party, clause class, conflict notion); tenant and document scope come from server-owned runtime context, exactly like every other tool.
- **Cap discipline: swap, don't add.** Each new tool enters a role's projection by replacing a same-role tool (per the plan's ≤7 cap), and the projection-consistency test of the `tool-capability-catalog` change enforces the subset invariant.

## Scope / Non-Goals

In scope: the three traversal tools, their envelope contracts, evidence-seam fusion, scope handling, registration under the catalog, and the cap-discipline tests.

Out of scope (explicit): the relationship edges themselves (the gate — retired plan Phase D / harden Phase 4); any change to extraction; graph write tools (P3: approval-gated writers are the harden change's 6.10); qna/portfolio *answering* (traversal returns structured results; narrative answers are qna's job via `mount-retrieval-graph`'s engine).

## Capabilities

### New Capabilities

- `portfolio-traversal-tools`: cross-document traversal over the typed relationship graph — thin Graphiti queries returning handle-bound envelopes, server-owned scope, gated on relationship edges existing.

### Modified Capabilities

- `agent-saul-delegation`: risk/compliance projections may include portfolio traversal tools under the ≤7 cap by replacing same-role tools; traversal results join the evidence-handle read path.

## Impact

Affected code: new traversal tool modules under `src/app/features/documents/tools/` (matching the existing per-tool module pattern), Graphiti query layer (read-only search/traverse calls), factory bindings with the swap rule, tests: scope denial, empty-edge behaviour (gate check), envelope bounds, cap discipline. Normative inputs: `PLAN-agent-tools.md` § 2 P2, § D2, § 6; `deep-research-bounded-subgraph` (handle run-scoping consumed); `tool-capability-catalog` (registration + projection tests consumed).
