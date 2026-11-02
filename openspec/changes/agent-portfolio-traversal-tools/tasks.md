> Change class: **M**. **Gated change — section 1 is the only work doable today; sections 2–4 are blocked on relationship edges existing in the knowledge graph (retired plan Phase D / harden Phase 4).** Each blocked task names its unblocking condition.

## 1. Contract scaffolding (doable now)

- [x] 1.1 Define the three traversal envelope models (identifiers, provenance, handles, coverage, status) and the gate probe (edge-existence check). Verify: models compile; the gate probe reports the current corpus state honestly. — Done: `src/app/features/documents/application/tools/portfolio.py` (3 frozen envelopes, status/bounds enforced by model validators, `relationship_edges_available` reusing `fetch_relationship_edges` read-only); 15 tests green in `tests/unit/features/documents/test_portfolio_traversal_gate.py`.
- [x] 1.2 Binding-time gate check in the graph factory path: binding a portfolio tool without edges fails naming the gate. Verify: a test drives the failure. — Done: `require_traversal_gate` raises `TraversalGateError` naming gate `relationship-edges`; no factory edits (integration stays gated); tests drive the no-edges and no-scope failures (`test_binding_gate_failure_names_the_gate`, `test_binding_gate_fails_closed_on_scopeless_probe`).

## 2. Traversal tools (blocked: relationship edges must exist)

- [ ] 2.1 `cross_document_obligations` — party → obligations across in-scope agreements. Verify: seeded-edges test returns handle-bound matches; scope-widening arguments cannot cross the authorized scope.
- [ ] 2.2 `find_conflicting_clauses` — conflicting clauses across documents in scope. Verify: seeded conflict fixture returns both sides with provenance and handles.
- [ ] 2.3 `party_obligation_map` — the party→obligation web. Verify: envelope bounds hold at high match counts; no raw clause text in the envelope.
- [ ] 2.4 Honest empty states: no-result queries return a distinct status, never an empty success. Verify: empty-graph and no-match tests.

## 3. Projection and cap discipline

- [ ] 3.1 Registration under the catalog with authority metadata (read-only, non-egress). Verify: registry + metadata agreement tests green.
- [ ] 3.2 Apply the swaps: each role gains a portfolio tool only with a same-role removal, recorded in this change's addendum. Verify: toolset-identity tests + projection-consistency suite green after the swap.

## 4. Gates and archive

- [ ] 4.1 `uv run ruff check --fix src/` · `uv run ty check src/` · `uv run pytest tests/unit/ -q` · `ast-grep scan src/` — green.
- [ ] 4.2 Sync deltas into `openspec/specs/portfolio-traversal-tools/` (new) and `openspec/specs/agent-saul-delegation/spec.md`; record the unblocking change in the archived entry.
